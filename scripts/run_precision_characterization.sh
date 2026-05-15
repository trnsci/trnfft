#!/usr/bin/env bash
#
# Run TestOzakiHQCharacterization on trn1 and print measured precision numbers.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_precision_characterization.sh [instance_type]
#
# Output: pytest stdout with the printed N=64/128/256 error tables.
# Typical run time: 5–10 min (NKI compile + 6 BF16 matmul runs per N).

set -euo pipefail

INSTANCE_TYPE="${1:-trn1}"
TAG="trnfft-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_precision_characterization.sh}"

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
  # Spot instances (one-time) cannot be stopped — terminate instead.
  LIFECYCLE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].InstanceLifecycle' --output text 2>/dev/null || echo "none")
  if [[ "$LIFECYCLE" == "spot" ]]; then
    echo "Terminating spot instance $INSTANCE_ID..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  else
    echo "Stopping $INSTANCE_ID..."
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  fi
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

# Wait for user_data (git clone + pip install) to complete on fresh instances.
echo "Waiting for /home/ubuntu/trnfft to exist (user_data may still be running)..."
for i in $(seq 1 40); do
  CHECK_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters '{"commands":["test -d /home/ubuntu/trnfft && echo READY || echo NOT_READY"],"executionTimeout":["15"]}' \
    --region "$REGION" \
    --output text --query 'Command.CommandId' 2>/dev/null) || true
  if [[ -n "$CHECK_ID" ]]; then
    sleep 5
    CHECK_OUT=$(aws ssm get-command-invocation \
      --command-id "$CHECK_ID" --instance-id "$INSTANCE_ID" \
      --region "$REGION" --query 'StandardOutputContent' --output text 2>/dev/null || echo "")
    if echo "$CHECK_OUT" | grep -q "READY"; then
      echo "Repo ready."
      break
    fi
  fi
  if [[ $i -eq 40 ]]; then
    echo "ERROR: /home/ubuntu/trnfft not found after 10 min; user_data may have failed" >&2
    exit 1
  fi
  echo "  [$i] waiting for user_data... (15s)"
  sleep 15
done

echo "Running TestOzakiHQCharacterization @ SHA=$SHA..."
TEST_SCRIPT="source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnfft && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  pytest tests/test_precision_modes.py::TestOzakiHQCharacterization \
    -v -m neuron -s --tb=short"

CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnfft precision characterization @ $SHA" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$TEST_SCRIPT\\\"\"],\"executionTimeout\":[\"1800\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Polling (up to 30 min)..."
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

if [[ "$STATUS" != "Success" ]]; then
  echo ""
  echo "=== STDERR ==="
  aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'StandardErrorContent' --output text
fi

echo ""
echo "=== Status: $STATUS ==="
[[ "$STATUS" == "Success" ]]
