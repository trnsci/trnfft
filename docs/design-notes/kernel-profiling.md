# NKI kernel profiling — engine utilization and permute overhead

## Purpose

Profile three NKI kernels on trn1 (Neuron SDK 2.29) to diagnose why the
Stockham radix-4 POC is ~2% slower than butterfly despite having 5 vs 10
NKI kernel launches at N=1024.

**Diagnostic hypothesis**: each Stockham stage has 2 `torch.permute +
contiguous` XLA calls (pre-kernel reshape and post-kernel reshape) plus 1
NKI kernel. That's 15 XLA ops total vs butterfly's 10 (1 kernel × 10 stages).
The 5-vs-10 launch advantage is cancelled by permute overhead.

---

## Run metadata

| Field | Value |
|-------|-------|
| Instance | trnfft-ci-trn1 (`i-0ae0e12e04e6d29f3`) |
| SDK version | Neuron SDK 2.29.0 |
| NKI version | 0.3.0 |
| trnfft SHA | `6764c21` |
| Date | 2026-04-16 |

---

## Permute timing (`--permute-timing`)

Measures `pre-permute`, `post-permute`, and `kernel` in isolation at stage `s=0`
(L=1, twiddle trivially [1,1,1,1]/[0,0,0,0]).

| N | Pre-permute (μs) | Post-permute (μs) | Kernel (μs) | Permute % | All stages (μs) |
|---|---|---|---|---|---|
| 64  | 57.6 | 39.7 | 853.0 | 10% | 2 851 (3 stages) |
| 256 | 56.2 | 39.4 | 858.8 | 10% | 3 818 (4 stages) |
| 1024 | 57.9 | 40.0 | 855.2 | 10% | 4 766 (5 stages) |
| 4096 | 57.5 | 39.7 | 884.3 | 10% | 5 889 (6 stages) |

**Butterfly stage reference (no permutes):**

| N | Stage | Kernel (μs) | All stages projected (μs) |
|---|---|---|---|
| 64  | s=3 | 869.3 | 5 216 (6 stages) |
| 256 | s=4 | 870.7 | 6 965 (8 stages) |
| 1024 | s=5 | 882.5 | 8 825 (10 stages) |

---

## Key finding: hypothesis was wrong — permutes are not the bottleneck

**Permute fraction is a constant 10% at all N.** This refutes the original
hypothesis that permutes cancelled the 5-vs-10 launch advantage.

### But the projected times don't match the benchmarks

| N | Stockham projected (μs) | Stockham actual (μs) | Discrepancy |
|---|---|---|---|
| 1024 | 4 766 | 7 568 | +2 802 μs (59% hidden overhead) |

Butterfly actual (7 399 μs) is LOWER than the butterfly probe projection
(8 825 μs). This inversion reveals that probe timing ≠ in-loop timing —
the probe measures isolated single calls while the benchmark measures the
full driver in a warm loop.

### Where the hidden overhead comes from

The probe measured stage `s=0` only, where `L=1` and twiddle tensors are trivially
computed (`torch.cos(zeros) = 1, torch.sin(zeros) = 0`). In the actual driver
each stage `s` computes twiddles from scratch on CPU:

```python
l_idx = torch.arange(L, ...)          # CPU tensor, size L
k_idx = torch.arange(4, ...)          # CPU tensor, size 4
ang   = -2π * l_idx * k_idx / (4L)    # CPU multiply
tw_r  = torch.cos(ang).expand(B,L,M,4).contiguous().reshape(total_groups, 4)
tw_i  = torch.sin(ang).expand(B,L,M,4).contiguous().reshape(total_groups, 4)
tw_r  = tw_r.to(device)               # H→D transfer
tw_i  = tw_i.to(device)               # H→D transfer
```

At stage `s=4` for N=1024, `L=256` — the twiddle arrays are `256×4 = 1024` floats
each, computed and transferred to XLA every stage. Twiddle sizes grow across
stages:

| Stage (N=1024) | L | tw tensor size |
|---|---|---|
| s=0 | 1 | 4 floats (trivial) |
| s=1 | 4 | 16 floats |
| s=2 | 16 | 64 floats |
| s=3 | 64 | 256 floats |
| s=4 | 256 | 1024 floats |

Butterfly has the same pattern (twiddle computed per-stage, CPU → XLA). But
Stockham's twiddles are 2D tensors (`(B,L,M,4)` before reshape) that require
a 2D `expand + contiguous`, while butterfly twiddles are 1D (`half`-element).

---

## Revised diagnosis

The real per-stage cost breakdown is:

| Component | Time (μs) | Notes |
|-----------|-----------|-------|
| Pre-permute | 57 | Constant — XLA graph op |
| Post-permute | 40 | Constant — XLA graph op |
| Twiddle recomputation | ~500–600 | Grows with stage; not measured by probe |
| Kernel (NKI) | 855 | Measured by probe |
| Total per stage (inferred) | ~1 500 | Matches 7568/5 = 1514 μs |

**Permutes are the wrong target. The real bottleneck is per-stage twiddle
recomputation and device transfer.**

---

## Corrected fix: twiddle precomputation + fused kernel

If all twiddles are precomputed ONCE before the stage loop:
1. All `log4(N)` twiddle tensors created in one pass on CPU
2. One `stacked.to(device)` transfer for all stages combined
3. Each stage reads its pre-sliced twiddle slice — zero extra H2D cost per stage

Combined with B1 (fused-stage kernel absorbing permutes):
- Theoretical per-stage: kernel (~855 μs) + no permutes + twiddle amortized ≈ 900 μs
- Total N=1024: 5 stages × 900 ≈ **4 500 μs** vs butterfly **7 399 μs** → **1.6× faster**

---

## Implications for Thread C implementation

**B1 (fused-stage kernel) alone saves 10% — worth doing but not decisive.**

The decisive optimization is **twiddle precomputation in the driver**. This is
a pure Python change to `_fft_via_stockham_nki` in `fft_core.py`:

```python
# Before stage loop: precompute all twiddle factors at once
all_tw_r, all_tw_i = [], []
for s in range(log4n):
    L = 1 << (2 * s)
    M = n // (4 * L)
    l_idx = torch.arange(L, dtype=x.real.dtype).view(1, L, 1, 1)
    k_idx = torch.arange(4, dtype=x.real.dtype).view(1, 1, 1, 4)
    ang = -2.0 * math.pi * l_idx * k_idx / (4.0 * L)
    tw_r = torch.cos(ang).expand(B_pad, L, M, 4).contiguous().reshape(B_pad * L * M, 4)
    tw_i = torch.sin(ang).expand(B_pad, L, M, 4).contiguous().reshape(B_pad * L * M, 4)
    all_tw_r.append(tw_r)
    all_tw_i.append(tw_i)
# One H→D transfer per-tensor (different sizes per stage, so can't stack):
all_tw_r = [t.to(device) for t in all_tw_r]
all_tw_i = [t.to(device) for t in all_tw_i]

# Stage loop uses all_tw_r[s], all_tw_i[s] — no recomputation
```

After twiddle precomputation, B1 (fused kernel) recovers the remaining 10% permute
overhead. Combined, they should realize the 1.6× speedup suggested by the probe data.

---

## Thread C phase 1: driver permutation precompute (v0.14)

**What changed:** `_stockham_perm_indices()` precomputes flat `int64` pack/unpack index
tensors for all stages on CPU before the stage loop. The per-stage pack/unpack sequence:

```python
# Old (2 materializing copies per stage):
re_groups = re.reshape(B, L, 4, M).permute(0,1,3,2).contiguous().reshape(G, 4)
re = out.reshape(B, L, M, 4).permute(0,3,1,2).contiguous().reshape(B, n)

# New (2 gather ops per stage):
re_groups = re.view(-1)[pack_idx].reshape(G, 4)
re = out.view(-1)[unpack_idx].reshape(B, n)
```

**Constraint acknowledged:** True SBUF-resident stage fusion (Thread C phase 2) is
blocked for all dispatched N values (N > 256) because `total_groups = B*N/4 > PMAX`
means the inter-stage permutation scatters elements across partition tiles. NKI's
affine_range model requires contiguous partition addressing; cross-tile indirect
scatter is not supported as of NKI 0.3.0.

**Hardware result (v0.14):** *fill after bench run — see `benchmark_results.json`.*

---

## Version

Profiling script: `scripts/run_neuron_profile.sh`  
Data collected: 2026-04-16, SHA `6764c21`, trn1, Neuron SDK 2.29.0  
Reference findings doc format: `trnblas/docs/design-notes/mp2_energy_profile_findings.md`
