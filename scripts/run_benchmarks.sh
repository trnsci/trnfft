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
# Design: separated setup + launch to avoid pip-install timeout busting the bench.
#
#   Phase 1a (setup, up to 60 min): SSM command runs git fetch/checkout + pip
#     install. Slow; allowed 3600s. Returns when SETUP_DONE is echoed.
#
#   Phase 1b (launch, < 10s): Three single-line SSM commands — rm stale files,
#     setsid launch (new OS session, SSM cannot track), echo LAUNCHED.
#     Avoids the bash `cmd1 && cmd2 & cmd3` → `(chain) & cmd3` precedence bug
#     that caused previous nohup/$! attempts to background the whole chain.
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
# Phase 1a: Setup — git fetch/checkout + pip install.
# Separated from the launch step so pip can take as long as it needs
# (up to 60 min) without interfering with the quick nohup launch.
# ---------------------------------------------------------------------------
echo "Setting up repo at SHA=$SHA (git fetch + pip install, up to 60 min)..."

RUNNER="/tmp/trnfft_bench_runner.sh"
SETUP_SCRIPT="set -e && \
  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnfft && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  printf '#!/bin/bash\nsource /opt/aws_neuronx_venv_pytorch_2_9/bin/activate\ncd /home/ubuntu/trnfft\nexec pytest benchmarks/bench_fft.py::TestFFT1DStockham benchmarks/bench_fft.py::TestFFT1DStockhamR8 benchmarks/bench_fft.py::TestFFT1DStockhamMixed benchmarks/bench_fft.py::TestFFT1DBF16 benchmarks/bench_fft.py::TestFFT1DOzaki -v --benchmark-only --benchmark-json=$REMOTE_JSON --benchmark-min-rounds=1 --benchmark-max-time=120 --tb=short >$REMOTE_LOG 2>&1\n' > $RUNNER && \
  chmod +x $RUNNER && \
  echo SETUP_DONE"

SETUP_CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnfft bench setup @ $SHA" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$SETUP_SCRIPT\\\"\"],\"executionTimeout\":[\"3600\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Setup command ID: $SETUP_CMD_ID"
for i in $(seq 1 240); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$SETUP_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 15
done

if [[ "$STATUS" != "Success" ]]; then
  echo "ERROR: Setup command failed ($STATUS)" >&2
  aws ssm get-command-invocation --command-id "$SETUP_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text >&2
  exit 1
fi
echo "Setup complete."

# ---------------------------------------------------------------------------
# Phase 1b: Launch — setsid creates a new OS session so SSM cannot track
# pytest's process group.  Three independent SSM commands (each is a single
# shell line) avoids the bash operator-precedence bug where
#   cmd1 && cmd2 & cmd3
# backgrounds the ENTIRE chain instead of just cmd2.
# ---------------------------------------------------------------------------
echo "Launching benchmark (setsid, new session)..."

# Build three-element commands JSON: rm stale files, setsid launch, confirm.
LAUNCH_CMDS="[
  \"sudo -u ubuntu rm -f $REMOTE_JSON $REMOTE_LOG\",
  \"sudo -u ubuntu setsid $RUNNER </dev/null >/dev/null 2>&1 &\",
  \"echo LAUNCHED\"
]"

LAUNCH_CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnfft bench launch @ $SHA" \
  --parameters "{\"commands\":$LAUNCH_CMDS,\"executionTimeout\":[\"120\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Launch command ID: $LAUNCH_CMD_ID"
for i in $(seq 1 30); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$LAUNCH_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 5
done

if [[ "$STATUS" != "Success" ]]; then
  echo "ERROR: Launch command failed ($STATUS)" >&2
  aws ssm get-command-invocation --command-id "$LAUNCH_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text >&2
  exit 1
fi

echo "Benchmark launched. Output: $REMOTE_LOG"
aws ssm get-command-invocation --command-id "$LAUNCH_CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text

# ---------------------------------------------------------------------------
# Phase 2: Poll for JSON completion with short-lived check commands.
# Checks every 60s; times out after 3 hours (180 checks).
# ---------------------------------------------------------------------------
echo "Polling for benchmark completion (up to 3 hours, checking every 60s)..."

# Two single-line commands — no embedded double quotes, safe in JSON parameters.
# Previous awk-based script had literal " from \"...\" expansion that corrupted
# the JSON, causing aws-cli to exit non-zero and set -e to kill the script.
CHECK_CMDS="[
  \"test -s $REMOTE_JSON && echo JSON_READY || echo NOT_DONE\",
  \"pgrep -f bench_fft && echo PYTEST_RUNNING || echo PYTEST_GONE\"
]"

DONE=0
for i in $(seq 1 180); do
  CHECK_CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "trnfft bench check $i" \
    --parameters "{\"commands\":$CHECK_CMDS,\"executionTimeout\":[\"30\"]}" \
    --region "$REGION" \
    --output text --query 'Command.CommandId' 2>/dev/null) || true

  if [[ -z "$CHECK_CMD_ID" ]]; then
    echo "  [$i] check dispatch failed, retrying in 30s"
    sleep 30
    continue
  fi

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
# Phase 3: Fetch + strip in one step (replaces former Phase 2.5 + Phase 3).
#
# All nine attempts at a separate strip-then-fetch approach produced 17997 bytes
# (SSM base64 truncation) because the strip never persisted the modified file —
# the stdout was empty every time, cause unknown.
#
# New design: a single SSM command runs a base64-encoded Python script that
#   1. reads /tmp/trnfft_bench.json (latin-1 to handle µs bytes from pytest-benchmark)
#   2. strips machine_info, commit_info, stats.data arrays IN MEMORY
#   3. writes base64(json.dumps(stripped)) directly to stdout
#
# SSM stdout receives ~300-500 chars (well under 24K).  No file write needed.
# Decoded locally: ~200-400 B valid JSON with 5 benchmark summaries.
#
# Python script (decoded from FETCH_B64):
#   import json, base64, sys
#   f = '/tmp/trnfft_bench.json'
#   raw = open(f, 'rb').read()
#   d = json.loads(raw.decode('latin-1'))
#   d.pop('machine_info', None)
#   d.pop('commit_info', None)
#   for b in d.get('benchmarks', []):
#       b.get('stats', {}).pop('data', None)
#   sys.stdout.buffer.write(base64.b64encode(json.dumps(d).encode('ascii')))
# ---------------------------------------------------------------------------
echo ""
echo "Fetching + stripping results JSON (combined strip+encode via SSM stdout)..."
FETCH_B64="aW1wb3J0IGpzb24sIGJhc2U2NCwgc3lzCmYgPSAnL3RtcC90cm5mZnRfYmVuY2guanNvbicKcmF3ID0gb3BlbihmLCAncmInKS5yZWFkKCkKZCA9IGpzb24ubG9hZHMocmF3LmRlY29kZSgnbGF0aW4tMScpKQpkLnBvcCgnbWFjaGluZV9pbmZvJywgTm9uZSkKZC5wb3AoJ2NvbW1pdF9pbmZvJywgTm9uZSkKZm9yIGIgaW4gZC5nZXQoJ2JlbmNobWFya3MnLCBbXSk6CiAgICBiLmdldCgnc3RhdHMnLCB7fSkucG9wKCdkYXRhJywgTm9uZSkKc3lzLnN0ZG91dC5idWZmZXIud3JpdGUoYmFzZTY0LmI2NGVuY29kZShqc29uLmR1bXBzKGQpLmVuY29kZSgnYXNjaWknKSkpCg=="
FETCH_CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "{\"commands\":[\"echo $FETCH_B64 | base64 -d | python3\"],\"executionTimeout\":[\"60\"]}" \
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
  echo "ERROR: Fetch+strip failed ($STATUS)" >&2
  aws ssm get-command-invocation --command-id "$FETCH_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text >&2
  exit 1
fi

aws ssm get-command-invocation --command-id "$FETCH_CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text \
  | tr -d '\n' | base64 --decode > "$LOCAL_OUT"

echo "Wrote $LOCAL_OUT ($(wc -c < "$LOCAL_OUT") bytes)"
echo ""
echo "Next: scripts/bench_to_md.py $LOCAL_OUT > docs/benchmarks_table.md"
