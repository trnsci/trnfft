# NKI kernel profiling — engine utilization and permute overhead

## Purpose

Profile three NKI kernels on trn1 (Neuron SDK 2.29) to diagnose why the
Stockham radix-4 POC is ~2% slower than butterfly despite having 5 vs 10
NKI kernel launches at N=1024.

**Diagnostic hypothesis**: each Stockham stage has 2 `torch.permute +
contiguous` XLA calls (pre-kernel reshape and post-kernel reshape) plus 1
NKI kernel. That's 15 XLA ops total vs butterfly's 10 (1 kernel × 10 stages).
The 5-vs-10 launch advantage is cancelled by permute overhead.

Data to collect:
1. **Permute fraction** — what % of per-stage wall time is `permute + contiguous`?
2. **Engine utilization** — is Vector Engine at the ceiling? Tensor Engine idle?
3. **Butterfly reference** — engine util for one butterfly stage vs one Stockham stage

If permute fraction > 20%, the fix is B1: fused-stage kernel absorbing
the index shuffle into NKI via strided `nl.load` (DMA stride pattern).

If Vector Engine ≥ 90% AND permutes are cheap, the fix is B2: Tensor-engine
twiddle (4×4 composite matrix per group).

---

## Profiling runs

*Placeholder — fill after hardware run.*

### Run metadata

| Field | Value |
|-------|-------|
| Instance | trnfft-ci-trn1 |
| SDK version | Neuron SDK 2.29.0 |
| NKI version | 0.3.0 |
| trnfft SHA | _fill_ |
| Date | _fill_ |

---

## Permute timing (`--permute-timing`)

*Placeholder — fill after `./scripts/run_neuron_profile.sh --permute-timing`*

| N | Pre-permute (μs) | Post-permute (μs) | Kernel (μs) | Permute % |
|---|---|---|---|---|
| 64 | | | | |
| 256 | | | | |
| 1024 | | | | |
| 4096 | | | | |

---

## Engine utilization — butterfly stage kernel

*Placeholder — fill after `./scripts/run_neuron_profile.sh --kernel butterfly`*

| Metric | Value |
|--------|-------|
| Vector Engine utilization | |
| Tensor Engine utilization | |
| DMA Engine utilization | |
| Wall time (μs) | |

---

## Engine utilization — Stockham radix-4 stage kernel

*Placeholder — fill after `./scripts/run_neuron_profile.sh --kernel stockham`*

| Metric | Value |
|--------|-------|
| Vector Engine utilization | |
| Tensor Engine utilization | |
| DMA Engine utilization | |
| Wall time (μs) | |

---

## Engine utilization — complex_gemm kernel (Tensor Engine reference)

*Placeholder — fill after `./scripts/run_neuron_profile.sh --kernel gemm`*

| Metric | Value |
|--------|-------|
| Vector Engine utilization | |
| Tensor Engine utilization | |
| DMA Engine utilization | |
| Wall time (μs) | |

---

## Interpretation

*Fill after all three profiles are in.*

**Decision**: B1 (fused-stage kernel) vs B2 (Tensor-engine twiddle)

---

## Version

Profiling script: `scripts/run_neuron_profile.sh`  
Reference findings doc format: `trnblas/docs/design-notes/mp2_energy_profile_findings.md`
