#!/usr/bin/env bash
#
# Run neuron-marked pytest tests on the trnfft CI instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_tests.sh [instance_type]
#
# Default instance_type is trn1 (looks for Name=trnfft-ci-trn1).
# Provision the instance with:
#   cd infra/terraform && terraform apply -var=vpc_id=... -var=subnet_id=...
#
# This script:
#   1. Starts the tagged instance (if stopped)
#   2. Waits for SSM agent
#   3. Runs `pytest tests/ -v -m neuron` via SSM send-command
#   4. Prints stdout/stderr
#   5. Stops the instance (always, even on failure)

set -euo pipefail

INSTANCE_TYPE="${1:-trn1}"
TAG="trnfft-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_neuron_tests.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  echo "Provision with: cd infra/terraform && terraform apply" >&2
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
echo "Waiting for SSM agent (polling)..."
for i in $(seq 1 30); do
  STATUS=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || echo "None")
  if [[ "$STATUS" == "Online" ]]; then
    echo "SSM agent online."
    break
  fi
  if [[ $i -eq 30 ]]; then
    echo "ERROR: SSM agent did not come online within 5 minutes" >&2
    exit 1
  fi
  sleep 10
done

echo "Sending test command (SHA=$SHA)..."
# Use a single bash -c invocation so we can rely on bash semantics (set -e,
# pipefail, $() substitution). SSM's default shell is sh, which doesn't
# support pipefail.
TEST_SCRIPT="source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnfft && \
  sudo -u ubuntu git fetch --all && \
  sudo -u ubuntu git checkout $SHA && \
  sudo -u ubuntu /opt/aws_neuronx_venv_pytorch_2_9/bin/pip install -e '/home/ubuntu/trnfft[dev]' --quiet && \
  sudo -u ubuntu /opt/aws_neuronx_venv_pytorch_2_9/bin/pytest tests/ -v -m neuron --tb=short"

CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnfft neuron tests @ $SHA" \
  --parameters "{\"commands\":[\"bash -c \\\"$TEST_SCRIPT\\\"\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (this may take several minutes)..."

# Poll instead of using `aws ssm wait command-executed` (it has a short
# built-in timeout that often fires before NKI compilation finishes).
STATUS="InProgress"
for i in $(seq 1 60); do
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
