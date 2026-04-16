#!/usr/bin/env bash
#
# Capture Neuron Profiler 2.0 traces of trnfft NKI kernels on the trnfft CI
# instance via SSM. Adapted from trnblas/scripts/run_neuron_profile.sh.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --probe
#     Phase A: confirm neuron-profile 2.0 API, list cached NEFFs
#
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --kernel butterfly
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --kernel stockham
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --kernel gemm
#     Phase B: profile one kernel — engine utilization + HBM bandwidth
#
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --permute-timing
#     Phase C: measure Stockham inter-stage permute overhead vs kernel time
#     (explains why the POC ties butterfly despite fewer stages)
#
# Output:
#   summary-text: per-engine utilization % printed to stdout
#   summary-json: raw metrics printed to stdout (head 300 lines)
#   profiles saved on instance under /home/ubuntu/profiles/trnfft-<kernel>-<sha>/
#
# ## Neuron Profiler 2.0 API (Neuron SDK 2.29)
#   neuron-profile capture -n <model.neff> -s <trace.ntff>
#   neuron-profile view   -n <model.neff> -s <trace.ntff> --output-format summary-text
#
# Double-base64 encoding bypasses all shell-quoting/heredoc-in-pipe-to-bash
# issues when sending scripts via SSM (same strategy as trnblas).

set -euo pipefail

KERNEL=""
PROBE=false
PERMUTE_TIMING=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)           PROBE=true; shift ;;
    --kernel)          KERNEL="$2"; shift 2 ;;
    --permute-timing)  PERMUTE_TIMING=true; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

TAG="trnfft-ci-trn1"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"
NP="/opt/aws/neuron/bin/neuron-profile"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_neuron_profile.sh}"

# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------
echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2; exit 1
fi
echo "Instance: $INSTANCE_ID"

cleanup() {
  local ec=$?
  echo ""
  echo "Stopping $INSTANCE_ID..."
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  exit "$ec"
}
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
if [[ "$STATE" == "stopping" ]]; then
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"; STATE=stopped
fi
if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi
echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
for _ in $(seq 1 60); do
  PING=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$PING" == "Online" ]] && break
  sleep 5
done
[[ "$PING" == "Online" ]] || { echo "ERROR: SSM agent not Online" >&2; exit 1; }

_run_ssm() {
  local comment="$1" body_b64="$2" wait_iters="${3:-60}" poll_interval="${4:-10}"
  local cmd_id
  cmd_id=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "$comment" \
    --parameters "commands=[\"printf '%s' $body_b64 | base64 -d | bash\"]" \
    --region "$REGION" \
    --output text --query 'Command.CommandId')
  echo "Command ID: $cmd_id"
  local status=InProgress
  for _ in $(seq 1 "$wait_iters"); do
    status=$(aws ssm get-command-invocation \
      --command-id "$cmd_id" --instance-id "$INSTANCE_ID" --region "$REGION" \
      --query 'Status' --output text 2>/dev/null || echo "InProgress")
    [[ "$status" != "InProgress" && "$status" != "Pending" ]] && break
    sleep "$poll_interval"
  done
  echo ""
  echo "=== STDOUT ==="
  aws ssm get-command-invocation --command-id "$cmd_id" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardOutputContent' --output text
  echo ""
  echo "=== STDERR ==="
  aws ssm get-command-invocation --command-id "$cmd_id" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text
  echo ""
  echo "=== Status: $status ==="
  [[ "$status" == "Success" ]]
}

# ---------------------------------------------------------------------------
# Phase A — probe
# ---------------------------------------------------------------------------
if [[ "$PROBE" == "true" ]]; then
  echo "Running Phase A probe (SHA=$SHA)..."
  PROBE_BODY=$(cat <<'PROBE_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
printf '%s\n' ==NP_VERSION==
$NP --version 2>&1 || true
printf '%s\n' ==CAPTURE_HELP==
$NP capture --help 2>&1 | head -30 || true
printf '%s\n' ==VIEW_OUTPUT_FORMAT==
$NP view --help 2>&1 | grep -i "output.format" || echo "NOT FOUND"
printf '%s\n' ==OLD_TRACES==
find /home/ubuntu/profiles -type f 2>/dev/null | head -20 || echo none
printf '%s\n' ==NEFF_CACHE==
find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -10 || echo none
PROBE_EOF
)
  B64=$(printf '%s' "$PROBE_BODY" | base64 | tr -d '\n')
  _run_ssm "trnfft neuron-profile probe @ $SHA" "$B64" 30 5
  exit 0
fi

# ---------------------------------------------------------------------------
# Helper: build a capture+profile body for a given Python warmup script
# ---------------------------------------------------------------------------
_build_capture_body() {
  local kernel_name="$1" py_b64="$2"
  local body
  body=$(cat <<'BODY_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1)
test -n "$NEURON_VENV" || { echo "ERROR: no Neuron venv" >&2; exit 1; }
PYTHON="$NEURON_VENV/bin/python"

cd /home/ubuntu
sudo -u ubuntu git -C /home/ubuntu/trnfft fetch --all --quiet
sudo -u ubuntu git -C /home/ubuntu/trnfft checkout __SHA__
sudo -u ubuntu env PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  "$PYTHON" -m pip install -e /home/ubuntu/trnfft[dev] --quiet

KNAME=__KERNEL_NAME__
PROFILE_DIR=/home/ubuntu/profiles/trnfft-${KNAME}-$(date +%s)
sudo -u ubuntu mkdir -p "$PROFILE_DIR"
chown -R ubuntu:ubuntu /home/ubuntu/profiles

printf '%s\n' ==STEP1_WRITE_WARMUP==
printf '%s' __PY_B64__ | base64 -d > /tmp/trnfft_warmup_${KNAME}.py
chown ubuntu:ubuntu /tmp/trnfft_warmup_${KNAME}.py
echo "Warmup script written."

printf '%s\n' ==STEP2_CLEAR_CACHE_AND_COMPILE==
rm -rf /var/tmp/neuron-compile-cache/* 2>/dev/null || true
sudo -u ubuntu env \
  PATH="$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin" \
  "$PYTHON" /tmp/trnfft_warmup_${KNAME}.py 2>&1

printf '%s\n' ==STEP3_FIND_NEFF==
NEFF=$(find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -1)
test -n "$NEFF" || { echo "ERROR: no model.neff after warmup" >&2; exit 1; }
echo "NEFF: $NEFF"
ls -lah "$NEFF"

printf '%s\n' ==STEP4_CAPTURE==
sudo -u ubuntu HOME=/home/ubuntu "$NP" capture \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" 2>&1

printf '%s\n' ==STEP5_SUMMARY_TEXT==
sudo -u ubuntu HOME=/home/ubuntu "$NP" view \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" \
  --output-format summary-text 2>&1

printf '%s\n' ==STEP6_SUMMARY_JSON==
sudo -u ubuntu HOME=/home/ubuntu "$NP" view \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" \
  --output-format summary-json 2>&1 | head -200

printf '%s\n' ==ARTIFACTS==
ls -laR "$PROFILE_DIR" 2>&1 | head -20
BODY_EOF
)
  body="${body//__SHA__/$SHA}"
  body="${body//__KERNEL_NAME__/$kernel_name}"
  body="${body//__PY_B64__/$py_b64}"
  printf '%s' "$body"
}

# ---------------------------------------------------------------------------
# Phase B — profile a specific kernel
# ---------------------------------------------------------------------------
_profile_kernel() {
  local name="$1"
  echo ""
  echo "========================================"
  echo "Profiling kernel: $name (SHA=$SHA)"
  echo "========================================"

  case "$name" in
    butterfly)
      PY_WARMUP=$(cat <<'PYEOF'
import sys, torch, math
sys.path.insert(0, '/home/ubuntu/trnfft')
import torch_xla
from trnfft import set_backend
from trnfft.nki.butterfly import butterfly_stage_kernel
set_backend("nki")
device = torch_xla.device()
# Profile butterfly at N=1024, stage=4 (total_groups=64 — near partition saturation)
B, N, stage = 1, 1024, 4
m = 1 << (stage + 1)
half = m >> 1
total_groups = B * (N // m)
x_re = torch.zeros(B, N, dtype=torch.float32).to(device)
x_im = torch.zeros(B, N, dtype=torch.float32).to(device)
tw_re = torch.ones(total_groups, half, dtype=torch.float32).to(device)
tw_im = torch.zeros(total_groups, half, dtype=torch.float32).to(device)
print(f"Compiling butterfly_stage_kernel N={N} stage={stage} total_groups={total_groups}", flush=True)
butterfly_stage_kernel(x_re, x_im, tw_re, tw_im, N, stage)
torch_xla.sync()
print("Done.", flush=True)
PYEOF
)
      ;;
    stockham)
      PY_WARMUP=$(cat <<'PYEOF'
import sys, torch, math
sys.path.insert(0, '/home/ubuntu/trnfft')
import torch_xla
from trnfft import set_backend
from trnfft.nki.stockham import stockham_radix4_stage_kernel
set_backend("nki")
device = torch_xla.device()
# Profile stockham stage at N=1024, s=0 (total_groups=256)
N, s = 1024, 0
B_pad = 1
L = 1 << (2 * s)
M = N // (4 * L)
total_groups = B_pad * L * M
x_re = torch.zeros(total_groups, 4, dtype=torch.float32).to(device)
x_im = torch.zeros(total_groups, 4, dtype=torch.float32).to(device)
tw_re = torch.ones(total_groups, 4, dtype=torch.float32).to(device)
tw_im = torch.zeros(total_groups, 4, dtype=torch.float32).to(device)
print(f"Compiling stockham_radix4_stage_kernel N={N} stage={s} total_groups={total_groups}", flush=True)
stockham_radix4_stage_kernel(x_re, x_im, tw_re, tw_im)
torch_xla.sync()
print("Done.", flush=True)
PYEOF
)
      ;;
    gemm)
      PY_WARMUP=$(cat <<'PYEOF'
import sys, torch
sys.path.insert(0, '/home/ubuntu/trnfft')
import torch_xla
from trnfft import set_backend, ComplexTensor
from trnfft.nki.dispatch import complex_gemm
set_backend("nki")
device = torch_xla.device()
N = 256
a = ComplexTensor(torch.zeros(N, N, dtype=torch.float32).to(device),
                  torch.zeros(N, N, dtype=torch.float32).to(device))
b = ComplexTensor(torch.zeros(N, N, dtype=torch.float32).to(device),
                  torch.zeros(N, N, dtype=torch.float32).to(device))
print(f"Compiling _complex_gemm_kernel N={N}", flush=True)
complex_gemm(a, b)
torch_xla.sync()
print("Done.", flush=True)
PYEOF
)
      ;;
    *)
      echo "ERROR: unknown kernel '$name' (choose: butterfly, stockham, gemm)" >&2
      exit 1
      ;;
  esac

  PY_B64=$(printf '%s' "$PY_WARMUP" | base64 | tr -d '\n')
  CAPTURE_BODY=$(_build_capture_body "$name" "$PY_B64")
  B64=$(printf '%s' "$CAPTURE_BODY" | base64 | tr -d '\n')
  _run_ssm "trnfft neuron-profile $name @ $SHA" "$B64" 120 30
}

if [[ -n "$KERNEL" ]]; then
  _profile_kernel "$KERNEL"
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase C — permute timing (no Neuron Profiler — Python wall-clock)
# ---------------------------------------------------------------------------
if [[ "$PERMUTE_TIMING" == "true" ]]; then
  echo "Running Phase C — permute timing (SHA=$SHA)..."

  PY_TIMING=$(cat <<'PYEOF'
"""
Measure the inter-stage permute overhead vs kernel time in the Stockham driver.

Hypothesis: each Stockham stage makes TWO permute+contiguous XLA calls
(pre-kernel and post-kernel) plus the NKI kernel. Butterfly has ONE kernel
call per stage and zero permutes. If permutes dominate, the 5-vs-10
launch-count advantage disappears.
"""
import sys, torch, math, time
sys.path.insert(0, '/home/ubuntu/trnfft')
import torch_xla
from trnfft import set_backend, ComplexTensor
from trnfft.nki.stockham import stockham_radix4_stage_kernel
from trnfft.nki.butterfly import butterfly_stage_kernel
set_backend("nki")
device = torch_xla.device()

def sync():
    torch_xla.sync()

def measure(fn, reps=20):
    fn()  # warmup
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
        sync()
    return (time.perf_counter() - t0) / reps * 1e6  # µs

print("="*60)
print("Stockham inter-stage permute vs kernel timing")
print("="*60)

for N in [64, 256, 1024, 4096]:
    if not (N > 0 and (N & (N-1)) == 0 and (N.bit_length() & 1) == 1):
        continue  # skip non-power-of-4
    B_pad = 1
    log4n = N.bit_length() // 2  # log4(N) = log2(N)/2

    # Stage 0 params (L=1, M=N//4, total_groups=N//4)
    s = 0
    L, M = 1, N // 4
    total_groups = B_pad * L * M

    re = torch.randn(B_pad, N, dtype=torch.float32).to(device)
    im = torch.zeros(B_pad, N, dtype=torch.float32).to(device)

    # Pre-twiddle groups
    re_4d = re.reshape(B_pad, L, 4, M)
    im_4d = im.reshape(B_pad, L, 4, M)
    re_groups = re_4d.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 4)
    im_groups = im_4d.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 4)
    tw_r = torch.ones(total_groups, 4, dtype=torch.float32).to(device)
    tw_i = torch.zeros(total_groups, 4, dtype=torch.float32).to(device)

    # Compile kernel first
    out_re, out_im = stockham_radix4_stage_kernel(re_groups, im_groups, tw_r, tw_i)
    sync()

    def run_pre_permute():
        re_4d_ = re.reshape(B_pad, L, 4, M)
        im_4d_ = im.reshape(B_pad, L, 4, M)
        _ = re_4d_.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 4)
        _ = im_4d_.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 4)

    def run_post_permute():
        out_4d = out_re.reshape(B_pad, L, M, 4)
        _ = out_4d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, N)

    def run_kernel():
        stockham_radix4_stage_kernel(re_groups, im_groups, tw_r, tw_i)

    t_pre  = measure(run_pre_permute)
    t_post = measure(run_post_permute)
    t_kern = measure(run_kernel)
    t_total_per_stage = t_pre + t_post + t_kern
    t_all_stages = t_total_per_stage * log4n
    permute_fraction = (t_pre + t_post) / t_total_per_stage * 100

    print(f"\nN={N:5d}  log4(N)={log4n}  total_groups(s=0)={total_groups}")
    print(f"  pre-permute  : {t_pre:7.1f} µs")
    print(f"  post-permute : {t_post:7.1f} µs")
    print(f"  kernel       : {t_kern:7.1f} µs")
    print(f"  per-stage    : {t_total_per_stage:7.1f} µs  ({permute_fraction:.0f}% permute)")
    print(f"  all {log4n} stages : {t_all_stages:7.1f} µs")

# Butterfly reference: time one butterfly stage at each N
print("\n" + "="*60)
print("Butterfly stage (no permutes) — reference")
print("="*60)
for N in [64, 256, 1024]:
    B = 1
    stage = int(math.log2(N)) // 2  # a mid-range stage
    m = 1 << (stage + 1)
    half = m >> 1
    total_groups = B * (N // m)
    x_re = torch.zeros(B, N, dtype=torch.float32).to(device)
    x_im = torch.zeros(B, N, dtype=torch.float32).to(device)
    tw_re = torch.ones(total_groups, half, dtype=torch.float32).to(device)
    tw_im = torch.zeros(total_groups, half, dtype=torch.float32).to(device)
    butterfly_stage_kernel(x_re, x_im, tw_re, tw_im, N, stage)  # compile
    sync()

    def run_bf():
        butterfly_stage_kernel(x_re, x_im, tw_re, tw_im, N, stage)

    t = measure(run_bf)
    log2n = int(math.log2(N))
    print(f"N={N:5d} stage={stage} total_groups={total_groups:4d}: {t:7.1f} µs/stage  ({t*log2n:.0f} µs all {log2n} stages)")

print("\nDone.")
PYEOF
)
  PY_B64=$(printf '%s' "$PY_TIMING" | base64 | tr -d '\n')

  TIMING_BODY=$(cat <<'TEOF'
set -euo pipefail
NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1)
test -n "$NEURON_VENV" || { echo "ERROR: no Neuron venv" >&2; exit 1; }
PYTHON="$NEURON_VENV/bin/python"
cd /home/ubuntu
sudo -u ubuntu git -C /home/ubuntu/trnfft fetch --all --quiet
sudo -u ubuntu git -C /home/ubuntu/trnfft checkout __SHA__
sudo -u ubuntu env PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  "$PYTHON" -m pip install -e /home/ubuntu/trnfft[dev] --quiet
printf '%s' __PY_B64__ | base64 -d > /tmp/trnfft_permute_timing.py
chown ubuntu:ubuntu /tmp/trnfft_permute_timing.py
sudo -u ubuntu env PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  "$PYTHON" /tmp/trnfft_permute_timing.py 2>&1
TEOF
)
  TIMING_BODY="${TIMING_BODY//__SHA__/$SHA}"
  TIMING_BODY="${TIMING_BODY//__PY_B64__/$PY_B64}"
  B64=$(printf '%s' "$TIMING_BODY" | base64 | tr -d '\n')
  _run_ssm "trnfft permute-timing @ $SHA" "$B64" 60 15
  exit 0
fi

# Default: profile all three kernels sequentially
echo "Profiling all three kernels (butterfly, stockham, gemm)..."
_profile_kernel butterfly
_profile_kernel stockham
_profile_kernel gemm
