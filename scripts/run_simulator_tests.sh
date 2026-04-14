#!/usr/bin/env bash
#
# Run NKI simulator-backed tests on the trn1 CI instance.
#
# Simulator bypasses torch_xla + NEFF compile: kernels run on CPU via
# nki.simulate(kernel)(numpy_args). Use for correctness / constraint
# iteration; hardware still owns perf numbers.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_simulator_tests.sh
#
# Same trap-stop pattern as run_neuron_tests.sh. Runs the nki_simulator-
# marked test suite with TRNFFT_USE_SIMULATOR=1 set in the SSM env.
#
# Still AWS-resident for now (the nki wheel is linux_x86_64 only + lives
# on the AWS pip index, not a common macOS target). The ubuntu-latest
# CI job is the GH-native equivalent.

set -euo pipefail

INSTANCE_TYPE="${INSTANCE_TYPE:-trn1}"
TAG="trnfft-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_simulator_tests.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,running,pending" \
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

if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi

echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
for _ in $(seq 1 60); do
  PING=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$PING" == "Online" ]] && break
  sleep 5
done
if [[ "$PING" != "Online" ]]; then
  echo "ERROR: SSM agent not Online after 5 minutes (last PingStatus=$PING)" >&2
  exit 1
fi

echo "Sending simulator test command (SHA=$SHA)..."
TEST_SCRIPT="source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnfft && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  TRNFFT_USE_SIMULATOR=1 pytest tests/ -v -m nki_simulator --tb=short"

CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnfft nki simulator tests @ $SHA" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$TEST_SCRIPT\\\"\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Polling (simulator is CPU-slow; may take 10+ min)..."

STATUS="InProgress"
for i in $(seq 1 120); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 15
done

echo ""
echo "=== STDOUT ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardOutputContent' --output text

echo ""
echo "=== STDERR ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardErrorContent' --output text

echo ""
echo "=== Status: $STATUS ==="
[[ "$STATUS" == "Success" ]]
