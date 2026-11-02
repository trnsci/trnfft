# Spherical harmonic transforms (`trnfft.sht`)

Scalar spherical harmonic transforms (SHT) of batches of real fields on the
sphere, built for spectral-transform models at L ≈ 500–1300 with hundreds to
thousands of fields per transform (trnsci/trnfft#148, trnsci/trnsci#111).

```python
import torch
import trnfft
from trnfft.sht import SHT

sht = SHT(lmax=1023)  # Gauss grid 1024 x 2048, fp32 tables (≈1.1 GB)
f = torch.randn(128, sht.nlat, sht.nlon)
alm = sht.forward(f)  # ComplexTensor (128, 524800): analysis
g = sht.inverse(alm)  # (128, 1024, 2048): synthesis
```

On Trainium, build the plan with `device="neuron"` (the tables are uploaded
once) and `trnfft.set_backend("nki")`; the latitude stage then runs on the
NKI kernels and the longitude FFTs on trnfft's NKI FFTs.

## Conventions

| | |
|---|---|
| harmonics | orthonormal, ∫\|Y_lm\|² dΩ = 1, Condon–Shortley phase (−1)^m; same as `ducc0.sht` and SHTns `sht_orthonormal` |
| coefficients | complex, m ≥ 0, triangular m-major order `index(l, m) = m(2L+3−m)/2 + (l−m)` (ducc0 / healpy / SHTns order), shape `(..., (L+1)(L+2)/2)` |
| `grid="gauss"` | Gauss–Legendre latitudes, `nlat = L + 1` by default |
| `grid="equiangular"` | θ_j = πj/(nlat−1) including both poles, Clenshaw–Curtis weights, `nlat = 2L + 2` by default |
| longitude | `nlon = 2L + 2` by default; any `nlon ≥ 2L + 1` |
| latitude order | index 0 at the north pole |

Both grids give exact quadrature for band-limited fields at their default
sizes, so `forward(inverse(a)) == a` up to rounding.

## Algorithm

1. **Longitude.** `trnfft.rfft` along each ring (or `torch.fft` with
   `fft="torch"`, for CPU baselines), the first L + 1 bins, scaled by the
   quadrature weight × 2π/nlon.
2. **Even/odd fold.** P_lm(−x) = (−1)^(l+m) P_lm(x), so the north and south
   rows are folded into sums and differences over the H = ⌈nlat/2⌉ northern
   latitudes. This splits each m into an even and an odd problem
   (p = 2m + q) of K_p ≈ (L − m + 1)/2 degrees, which halves the flops and the
   table.
3. **Latitude.** For every problem, a GEMM against the table P_p
   (K_p × H): `a_p = P_p · X_p` (analysis), `Y_p = P_pᵀ · a_p` (synthesis).
   The 2F columns are the real and imaginary parts of all fields: the table
   is real, so complex coefficients are carried as real column pairs and no
   complex arithmetic reaches the device.
4. **Layout.** One permute each way between the field-major user layout and
   the m-major GEMM layout. Inside an m block the degrees of one parity are
   every other coefficient row; the kernels read and write them as stride-2
   DMA access patterns, not gathers.

### The NKI latitude kernels

`trnfft/nki/sht.py` holds `sht_legendre_forward_kernel` and
`sht_legendre_inverse_kernel`. One launch runs all 2(L + 1) ragged
problems of a transform (or a few launches, see below): a grouped GEMM with
static shapes.

- **Stationary table.** For each problem, the live 128 × 128 table tiles are
  loaded into SBUF once. Every moving tile of up to 512 columns then streams
  through the Tensor Engine into fp32 PSUM. The table crosses HBM once per
  transform, whatever F is, and the arithmetic intensity is about L flops per
  field element. With F ≥ 128 that makes the latitude stage compute-bound.
- **Transposes.** The analysis kernel needs P_pᵀ as the stationary operand.
  Each tile is read with a plain `nl.load` and transposed on the Tensor
  Engine with `nisa.nc_transpose`, once per problem rather than once per
  field tile. No kernel uses `nl.load_transpose2d` (trnfft#97,
  aws-neuron-sdk#1452).
- **Polar tiles.** Near the poles P_lm underflows for large m. Table entries
  below `skip_threshold` (default 1e-30) times the largest |P_lm| of their m
  are set to zero. A tile that is then all zero is never loaded or
  multiplied, and an output tile with no live tile is stored as zeros. The
  static mask travels in the kernel's `plan` string argument. That puts it in
  the NKI trace-cache key; a value captured in a closure or a global would not
  be (aws-neuron-sdk#1454). At the Gauss grid with the default threshold this
  skips 3.9 % of the tiles at L = 511 (0.7 % of the flops) and 11.3 % at
  L = 1023 (9.4 % of the flops). The skipped share grows with L.
- **Both physical cores.** The kernels launch gridless. At LNC=2
  (`num_programs() == 2`) each program takes half of the 2F columns. Both
  programs therefore run identical shapes and masks, and each holds its own
  copy of the table tiles in its SBUF.
- **Tail tiles.** K_p, H and the column count need not be multiples of
  128, 128 and 512. The last partial tile of each is traced at its own static
  size; `tests/test_sht_nki_sim.py` covers L ∈ {0, 7, 40, 130, 257, 300}.
- **Launch chunks.** The m range is cut into launches of at most
  `TRNFFT_SHT_TILES_PER_LAUNCH` live tiles (default 8192; 0 = one launch), to
  bound each traced program. That gives one launch at L = 511 and three at
  L = 1023. On the CPU the NKI frontend traces the single-launch L = 1023
  kernel in about 30 s.

`legendre="torch"` replaces the kernels with a loop of one matmul per
problem: the per-m aten loop of trnsci/trnsci#111, 2(L + 1) launches. It is
kept as the CPU path and as the device comparison row. trnsci/trnblas#98
(grouped GEMM) is the general-purpose form of what the kernels do here.

## Sizes

| L | grid | tables (fp32) | latitude flops per transform, F = 1024 (after skipping) |
|---|---|---|---|
| 511 | 512 × 1024 | 0.13 GB | 1.4 × 10¹¹ |
| 1023 | 1024 × 2048 | 1.07 GB | 1.0 × 10¹² |

The device working set at L = 1023 and F = 1024 is about three 8.6 GB
arrays: the grid, the longitude coefficients, and the FFT temporaries.
`trnfft.rfft` computes the full complex FFT (trnfft#124). That is more than
the 24 GiB of one logical core at LNC=2, so `scripts/bench_sht.py
--field-chunk 256` runs F in chunks.

## Accuracy

**Tables.** The tables are computed on the host in fp64 by the standard
three-term recurrence in l. P_mm is started in log space as an fp64 mantissa
times 2^e, and the mantissa is rescaled whenever it passes 2^256. sin^m θ at
m ≈ 1000 near the pole is far below the fp64 range, but P_lm grows back by
many orders of magnitude before l reaches L, so a plain fp64 start would lose
those values. The fp64 tables are checked in three ways:

- against a closed form (lgamma) for P_mm at m up to 1022;
- for orthonormality under the Gauss quadrature, to 1e-12, at m = 1000 and
  1023 with L = 1023;
- against ducc0, SHTns and a direct `scipy.special.sph_harm_y` quadrature, to
  1e-12–1e-13.

The tables are rounded once to fp32. That costs a relative error of at most
u = 2⁻²⁴ ≈ 6e-8 per entry, and the polar drop is far below that.

**Transforms.** The table: relative spectral ℓ2 error against fp64 ducc0 on
random band-limited fields (Gauss grid; measured on the CPU with fp32
tables, fp32 FFT and fp32 GEMM accumulation, which is what the device
computes):

| L | fp64 path | fp32 tables only (fp64 arithmetic) | forward, fp32 | inverse, fp32 | round trip, fp32 |
|---|---|---|---|---|---|
| 127 | 1.3e-14 | 2.3e-8 | 1.5e-7 | 1.5e-7 | – |
| 511 | 4.4e-14 | 2.3e-8 | 2.6e-7 | 2.6e-7 | 3.6e-7 |
| 1023 | 8.8e-14 | 2.3e-8 | 3.6e-7 | 3.5e-7 | 4.9e-7 |

- **Table rounding** contributes about u/√3 ≈ 2.3e-8 at every L. Random
  rounding errors in an orthogonal transform do not accumulate.
- **The fp32 arithmetic dominates.** That is the FFT's log₂(nlon) stages and
  the K-term sums of the latitude GEMM. It grows slowly with L, from
  1.5e-7 to 3.6e-7 over an 8× range of L. That is consistent with
  u·√(log₂ N + c·K) growth for random rounding, not the worst-case u·K
  bound.

**Tolerance.** The tests require **≤ 1e-6** relative error (spectral ℓ2) per
direction, and ≤ 2e-6 for a round trip, which is two transforms. 1e-6 is
about 17 u. That leaves headroom above the measured 1.5–3.6e-7 for:

- other inputs;
- the device's accumulation order;
- extrapolation to L = 1279 (the trnsci/trnsci#111 target; the trend gives
  about 4e-7).

It is still tight enough that a wrong tile, a dropped tail or an fp32 range
problem fails it by orders of magnitude.

On the device, the fp32 PSUM accumulates in the same precision as the CPU
GEMM. The device rows of `scripts/bench_sht.py` report the measured errors
next to the timings.

## Validation

- `tests/test_sht.py` (CPU, part of `pytest -m "not neuron"`):
    - fp64 results against ducc0 and SHTns (both `pip install`-able), and
      against a direct `scipy.special.sph_harm_y` quadrature;
    - fp32 results within 1e-6 of ducc0 at L = 31, 127 and, with
      `--run-slow`, 511;
    - grid quadrature exactness;
    - the high-m tables;
    - mask exactness, the plan encoding and the stage structure.
- `tests/test_sht_nki_sim.py` (`TRNFFT_USE_SIMULATOR=1`): the kernels under
  `nki.simulate` at 1 and 2 programs, against the torch loop. The cases cover
  tails, odd nlat, chunked launches, and a poisoned-table test: masked tiles
  hold NaN, and the result stays finite and exact.
- `tests/test_nki_compile.py` (`sht-*`): the NKI frontend's MLIR verifier on
  trn1, trn2 at LNC=1 and trn2 at LNC=2.
- `tests/test_sht.py::test_device_kernels_match_fp64` (`-m neuron`): the
  device kernels within 1e-6 of fp64 at L = 127, 511 and 1023.

## Benchmark

`scripts/bench_sht.py` compares these rows on the same box:

- `trnfft.sht` on the CPU, at 2 threads and at all vCPUs;
- ducc0 and SHTns, at the same thread counts;
- with `--device`, the device with the per-m aten loop (`device-aten`) and
  with the NKI kernels (`device-nki`).

Each row reports cold and warm times, the stage split (`--stages`), and the
error against fp64.

## Not yet

- Vector (spheroidal/toroidal) transforms. They are the next step of
  trnsci/trnsci#111 milestone 1.
- Longitude sizes such as 1280 and 2560 without chirp-z (trnfft#149). They
  work today through trnfft's Bluestein path, but are slower.
- Half-cost real FFTs (trnfft#124).
