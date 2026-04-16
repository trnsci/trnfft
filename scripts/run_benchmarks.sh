#!/usr/bin/env bash
#
# Run trnfft benchmarks on the trnfft CI instance and pull results back.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_benchmarks.sh [instance_type]
#
# Default instance_type is trn1. Produces:
#   - benchmark_results.json (raw pytest-benchmark output)
#
# Design: two-phase to avoid local timeout killing the benchmark.
#
#   Phase 1 (submit, ~2 min): SSM command installs deps and starts pytest
#     as a nohup background process. Returns as soon as pytest is launched.
#
#   Phase 2 (poll, up to 3 h): Short-lived SSM "check" commands poll for
#     /tmp/trnfft_bench.json to become non-empty (pytest-benchmark writes it
#     on completion). No long-running SSM command — each check is < 5 sec.
#
#   Phase 3 (fetch): Pull the JSON via base64 SSM command.
#
#   Phase 4 (stop): Stop the instance.

set -euo pipefail

INSTANCE_TYPE="${1:-trn1}"
TAG="trnfft-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"
LOCAL_OUT="${BENCH_OUTPUT:-benchmark_results.json}"
REMOTE_JSON="/tmp/trnfft_bench.json"
REMOTE_LOG="/tmp/trnfft_bench.log"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_benchmarks.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  exit 1
fi
echo "Instance: $INSTANCE_ID"

cleanup() {
  local exit_code=$?
  echo ""
  echo "Stopping $INSTANCE_ID..."
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  exit "$exit_code"
}
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
if [[ "$STATE" == "stopping" ]]; then
  echo "Instance is stopping; waiting..."
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
  STATE="stopped"
fi
if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi
echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
for i in $(seq 1 30); do
  STATUS=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || echo "None")
  if [[ "$STATUS" == "Online" ]]; then echo "SSM agent online."; break; fi
  if [[ $i -eq 30 ]]; then echo "ERROR: SSM agent not online" >&2; exit 1; fi
  sleep 10
done

# ---------------------------------------------------------------------------
# Phase 1: Submit benchmark as a background nohup process.
# The SSM command installs deps, removes any stale JSON, launches pytest in
# the background, and exits immediately. No long-running SSM command.
# ---------------------------------------------------------------------------
echo "Submitting benchmark (SHA=$SHA) as background process..."

SUBMIT_SCRIPT="set -e && \
  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnfft && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  rm -f $REMOTE_JSON $REMOTE_LOG && \
  nohup pytest benchmarks/bench_fft.py --benchmark-only --benchmark-json=$REMOTE_JSON --tb=short \
    > $REMOTE_LOG 2>&1 & \
  echo SUBMITTED_PID=\$!"

SUBMIT_CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnfft bench submit @ $SHA" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$SUBMIT_SCRIPT\\\"\"],\"executionTimeout\":[\"600\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Submit command ID: $SUBMIT_CMD_ID"
for i in $(seq 1 20); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$SUBMIT_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 15
done

if [[ "$STATUS" != "Success" ]]; then
  echo "ERROR: Submit command failed ($STATUS)" >&2
  aws ssm get-command-invocation --command-id "$SUBMIT_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text >&2
  exit 1
fi

echo "Benchmark launched. Output: $REMOTE_LOG"
aws ssm get-command-invocation --command-id "$SUBMIT_CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text

# ---------------------------------------------------------------------------
# Phase 2: Poll for JSON completion with short-lived check commands.
# Checks every 60s; times out after 3 hours (180 checks).
# ---------------------------------------------------------------------------
echo "Polling for benchmark completion (up to 3 hours, checking every 60s)..."

CHECK_SCRIPT="set -e && \
  if [ -s $REMOTE_JSON ]; then \
    echo JSON_READY && \
    wc -c < $REMOTE_JSON; \
  else \
    echo NOT_DONE && \
    ps aux | grep pytest | grep -v grep | awk '{print \"PYTEST_RUNNING PID=\"\$2}' || true; \
  fi"

DONE=0
for i in $(seq 1 180); do
  CHECK_CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "trnfft bench check $i" \
    --parameters "{\"commands\":[\"bash -c \\\"$CHECK_SCRIPT\\\"\"],\"executionTimeout\":[\"30\"]}" \
    --region "$REGION" \
    --output text --query 'Command.CommandId' 2>/dev/null)

  for j in $(seq 1 6); do
    STATUS=$(aws ssm get-command-invocation \
      --command-id "$CHECK_CMD_ID" \
      --instance-id "$INSTANCE_ID" \
      --region "$REGION" \
      --query 'Status' --output text 2>/dev/null || echo "Unknown")
    [ "$STATUS" = "Success" ] && break
    [ "$STATUS" = "Failed" ] && break
    sleep 5
  done

  CHECK_OUT=$(aws ssm get-command-invocation \
    --command-id "$CHECK_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'StandardOutputContent' --output text 2>/dev/null || echo "")

  if echo "$CHECK_OUT" | grep -q "JSON_READY"; then
    echo "  [$i] JSON ready ($(echo "$CHECK_OUT" | grep -oE '[0-9]+' | tail -1) bytes)"
    DONE=1
    break
  else
    PYTEST_STATUS=$(echo "$CHECK_OUT" | grep "PYTEST_RUNNING" || echo "pytest gone?")
    echo "  [$i] Not done yet. $PYTEST_STATUS"
  fi
  sleep 60
done

if [[ "$DONE" -ne 1 ]]; then
  echo "ERROR: Benchmark timed out after 3 hours" >&2
  # Print last lines of log for diagnosis
  aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "{\"commands\":[\"tail -50 $REMOTE_LOG\"],\"executionTimeout\":[\"15\"]}" \
    --region "$REGION" >/dev/null
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 3: Fetch the JSON.
# ---------------------------------------------------------------------------
echo ""
echo "Pulling results JSON via SSM (base64)..."
FETCH_CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "{\"commands\":[\"base64 -w0 $REMOTE_JSON\"],\"executionTimeout\":[\"60\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

for i in $(seq 1 12); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$FETCH_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 5
done

if [[ "$STATUS" != "Success" ]]; then
  echo "ERROR: Could not fetch JSON. Status: $STATUS" >&2
  exit 1
fi

aws ssm get-command-invocation --command-id "$FETCH_CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text \
  | tr -d '\n' | base64 --decode > "$LOCAL_OUT"

echo "Wrote $LOCAL_OUT ($(wc -c < "$LOCAL_OUT") bytes)"
echo ""
echo "Next: scripts/bench_to_md.py $LOCAL_OUT > docs/benchmarks_table.md"
