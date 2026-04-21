#!/usr/bin/env bash
#
# Collect NKI compiler artifacts for nl.load_transpose2d bug report.
# Runs reproduce_nl_load_transpose2d.py on trn1 with --pipeline compile SaveTemps,
# then fetches log-neuron-cc.txt and tars the NKI artifacts back via SSM stdout.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/collect_nki_artifacts.sh [instance_type]
#
# Output:
#   nki_artifacts.tar.gz  — neuronx-cc workdir + /tmp/nki_* from the failed compilation
#   nki_compiler_log.txt  — log-neuron-cc.txt content

set -euo pipefail

INSTANCE_TYPE="${1:-trn1}"
TAG="trnfft-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"
ARTIFACTS_OUT="nki_artifacts.tar.gz"
LOG_OUT="nki_compiler_log.txt"
REMOTE_SCRIPT="/tmp/reproduce_nl_load_transpose2d.py"
REMOTE_WORKDIR="/tmp/nki_artifact_collection"

: "${AWS_PROFILE:?Set AWS_PROFILE}"

echo "Looking up instance $TAG..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text --region "$REGION")

[[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]] && { echo "ERROR: no instance $TAG" >&2; exit 1; }
echo "Instance: $INSTANCE_ID"

cleanup() { aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null; }
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
[[ "$STATE" == "stopping" ]] && {
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
  STATE="stopped"
}
[[ "$STATE" == "stopped" ]] && aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM..."
for i in $(seq 1 30); do
  STATUS=$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo "None")
  [[ "$STATUS" == "Online" ]] && { echo "SSM online."; break; }
  [[ $i -eq 30 ]] && { echo "ERROR: SSM not online" >&2; exit 1; }
  sleep 10
done

# ---------------------------------------------------------------------------
# Phase 1: Setup — install repo at current SHA
# ---------------------------------------------------------------------------
echo "Setting up repo at SHA=$SHA..."
SETUP_SCRIPT="set -e && \
  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnfft && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  echo SETUP_DONE"

SETUP_CMD=$(aws ssm send-command --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$SETUP_SCRIPT\\\"\"],\"executionTimeout\":[\"3600\"]}" \
  --region "$REGION" --output text --query 'Command.CommandId')

for i in $(seq 1 240); do
  S=$(aws ssm get-command-invocation --command-id "$SETUP_CMD" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'Status' --output text 2>/dev/null || echo Unknown)
  case "$S" in Success|Failed|TimedOut|Cancelled) break ;; esac
  sleep 15
done
[[ "$S" != "Success" ]] && { echo "ERROR: setup failed ($S)" >&2; exit 1; }
echo "Setup done."

# ---------------------------------------------------------------------------
# Phase 2: Run reproducer with SaveTemps + NKI artifact collection
# ---------------------------------------------------------------------------
echo "Running reproducer with --pipeline compile SaveTemps..."

# Build the run command — sets all SaveTemps env vars, runs script, then
# finds the neuronx-cc workdir and nki artifacts.
RUN_CMD="sudo -u ubuntu bash -c \"\
  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnfft && \
  mkdir -p $REMOTE_WORKDIR && \
  export NEURON_CC_FLAGS='--pipeline compile SaveTemps' && \
  export NKI_COMPILE_CACHE_URL='$REMOTE_WORKDIR/nki_cache' && \
  export NEURON_CC_FLAGS=\\\"\$NEURON_CC_FLAGS --cache_dir=$REMOTE_WORKDIR/cc_cache\\\" && \
  python3 scripts/reproduce_nl_load_transpose2d.py > $REMOTE_WORKDIR/run.log 2>&1 || true && \
  echo RUN_COMPLETE\""

RUN_CMD_ID=$(aws ssm send-command --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "{\"commands\":[\"$RUN_CMD\"],\"executionTimeout\":[\"300\"]}" \
  --region "$REGION" --output text --query 'Command.CommandId')

for i in $(seq 1 30); do
  S=$(aws ssm get-command-invocation --command-id "$RUN_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'Status' --output text 2>/dev/null || echo Unknown)
  case "$S" in Success|Failed|TimedOut|Cancelled) break ;; esac
  sleep 10
done
echo "Run command status: $S"

# ---------------------------------------------------------------------------
# Phase 3: Collect artifacts — tar up workdir + /tmp/nki_* + run log
# ---------------------------------------------------------------------------
echo "Collecting artifacts..."

# Python script: find artifacts, tar them, base64-encode for SSM stdout
COLLECT_B64=$(python3 -c "
import base64
script = '''
import os, tarfile, base64, sys, glob

workdir = \"$REMOTE_WORKDIR\"
out_tar = \"/tmp/nki_artifacts_out.tar.gz\"

with tarfile.open(out_tar, \"w:gz\") as tar:
    # Run log
    run_log = os.path.join(workdir, \"run.log\")
    if os.path.exists(run_log):
        tar.add(run_log, arcname=\"run.log\")

    # NKI cache
    nki_cache = os.path.join(workdir, \"nki_cache\")
    if os.path.isdir(nki_cache):
        tar.add(nki_cache, arcname=\"nki_cache\")

    # neuronx-cc workdirs (contain log-neuron-cc.txt, sg00/ NKI artifacts)
    cc_cache = os.path.join(workdir, \"cc_cache\")
    if os.path.isdir(cc_cache):
        tar.add(cc_cache, arcname=\"cc_cache\")

    # Any /tmp/nki_* files (NKI JSON artifacts)
    for p in glob.glob(\"/tmp/nki_*.json\"):
        tar.add(p, arcname=os.path.join(\"tmp_nki\", os.path.basename(p)))

    # neuronx-cc temp work dirs
    for p in glob.glob(\"/tmp/MODULE_*\") + glob.glob(\"/tmp/neuronx-cc*\"):
        try:
            tar.add(p, arcname=os.path.join(\"tmp_cc\", os.path.basename(p)))
        except Exception:
            pass

sys.stdout.buffer.write(base64.b64encode(open(out_tar, \"rb\").read()))
'''
print(base64.b64encode(script.encode()).decode())
")

COLLECT_CMD_ID=$(aws ssm send-command --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "{\"commands\":[\"echo $COLLECT_B64 | base64 -d | python3\"],\"executionTimeout\":[\"120\"]}" \
  --region "$REGION" --output text --query 'Command.CommandId')

for i in $(seq 1 24); do
  S=$(aws ssm get-command-invocation --command-id "$COLLECT_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'Status' --output text 2>/dev/null || echo Unknown)
  case "$S" in Success|Failed|TimedOut|Cancelled) break ;; esac
  sleep 5
done

if [[ "$S" != "Success" ]]; then
  echo "ERROR: artifact collection failed ($S)" >&2
  aws ssm get-command-invocation --command-id "$COLLECT_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text >&2
  exit 1
fi

# Fetch and decode the tar
aws ssm get-command-invocation --command-id "$COLLECT_CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text \
  | tr -d '\n' | base64 --decode > "$ARTIFACTS_OUT"

echo "Wrote $ARTIFACTS_OUT ($(wc -c < "$ARTIFACTS_OUT") bytes)"

# Also pull out the run log directly for quick inspection
aws ssm send-command --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "{\"commands\":[\"cat $REMOTE_WORKDIR/run.log\"],\"executionTimeout\":[\"15\"]}" \
  --region "$REGION" --output text --query 'Command.CommandId' > /tmp/_logcmd
sleep 5
LOG_CMD=$(cat /tmp/_logcmd)
aws ssm get-command-invocation --command-id "$LOG_CMD" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text > "$LOG_OUT" 2>/dev/null || true

echo "Wrote $LOG_OUT"
echo ""
echo "=== Run output ==="
cat "$LOG_OUT"
echo ""
echo "Next steps:"
echo "  tar tzf $ARTIFACTS_OUT   # inspect artifact contents"
echo "  tar xzf $ARTIFACTS_OUT   # extract"
echo "  # share $ARTIFACTS_OUT with AWS Neuron team"
