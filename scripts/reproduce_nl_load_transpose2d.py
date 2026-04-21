"""
Minimal reproducer: nl.load_transpose2d from kernel-local shared_hbm scratch.

Issue: nl.load_transpose2d silently accepts a kernel-local nl.ndarray(buffer=
nl.shared_hbm) in nki.simulate but fails NEFF compilation on trn1 hardware
without a useful error message.

Expected: either
  (a) NEFF compilation succeeds (constraint does not exist), or
  (b) Compilation fails with an error naming the constraint — currently it
      fails silently (no message explaining that load_transpose2d requires a
      function-argument HBM tensor, not a kernel-local allocation).

Context: encountered while implementing a two-phase Stockham radix-8 FFT
kernel (trnfft v0.15, commit 307b89f). Simulator passed; hardware failed.
Workaround: move twiddle multiply to the PyTorch driver and pass pre-twiddled
data as an explicit function argument so load_transpose2d sees external HBM.

NKI version: 0.3.0  |  SDK: 2.29  |  Target: trn1
"""

import os
import sys

import torch
import torch_xla

import nki
import nki.isa as nisa
import nki.language as nl

# ---------------------------------------------------------------------------
# Failing kernel: load_transpose2d from kernel-local shared_hbm scratch
# ---------------------------------------------------------------------------


@nki.jit
def kernel_with_local_scratch(x_in, w_mat):
    """
    Phase 1: write x_in → kernel-local scratch (nl.shared_hbm).
    Phase 2: nl.load_transpose2d from that scratch → nc_matmul.

    Passes nki.simulate; fails NEFF compilation on trn1 hardware.
    """
    M, K = x_in.shape
    _, N = w_mat.shape

    out = nl.ndarray((M, N), dtype=x_in.dtype, buffer=nl.shared_hbm)

    # Kernel-local scratch — this is the problematic allocation.
    scratch = nl.ndarray((M, K), dtype=x_in.dtype, buffer=nl.shared_hbm)

    PMAX = 128
    n_tiles = M // PMAX

    # Phase 1: write to kernel-local scratch (no issue here).
    for p in nl.affine_range(n_tiles):
        p_off = p * PMAX
        tile = nl.load(x_in[p_off : p_off + PMAX, :])
        nl.store(scratch[p_off : p_off + PMAX, :], tile)

    # Load W matrix (external arg — this is fine).
    w = nl.load(w_mat[:K, :N])

    # Phase 2: load_transpose2d from kernel-local scratch.
    # This is the line that fails NEFF compilation.
    for m in nl.affine_range(n_tiles):
        m_off = m * PMAX
        psum = nl.zeros((PMAX, N), dtype=nl.float32, buffer=nl.psum)

        # FAILS HERE on hardware: scratch is kernel-local, not a function arg.
        x_t = nl.load_transpose2d(scratch[m_off : m_off + PMAX, :K])

        nisa.nc_matmul(dst=psum, stationary=x_t, moving=w, accumulate=True)

        result = nl.ndarray((PMAX, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=psum)
        nl.store(out[m_off : m_off + PMAX, :], result)

    return out


# ---------------------------------------------------------------------------
# Working kernel: load_transpose2d from function-argument HBM (workaround)
# ---------------------------------------------------------------------------


@nki.jit
def kernel_with_external_arg(x_in, w_mat):
    """
    Same computation but x_in is passed directly as a function argument.
    nl.load_transpose2d from a function-argument HBM tensor works correctly.
    """
    M, K = x_in.shape
    _, N = w_mat.shape

    out = nl.ndarray((M, N), dtype=x_in.dtype, buffer=nl.shared_hbm)
    PMAX = 128
    n_tiles = M // PMAX

    w = nl.load(w_mat[:K, :N])

    for m in nl.affine_range(n_tiles):
        m_off = m * PMAX
        psum = nl.zeros((PMAX, N), dtype=nl.float32, buffer=nl.psum)

        # Works: x_in is a function argument, not a kernel-local allocation.
        x_t = nl.load_transpose2d(x_in[m_off : m_off + PMAX, :K])

        nisa.nc_matmul(dst=psum, stationary=x_t, moving=w, accumulate=True)

        result = nl.ndarray((PMAX, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=psum)
        nl.store(out[m_off : m_off + PMAX, :], result)

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    M, K, N = 512, 8, 8  # total_groups=512, radix-8 DFT stage
    device = torch_xla.device()

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float32).to(device)
    w = torch.randn(K, N, dtype=torch.float32).to(device)

    print(f"Target device: {device}")
    print(f"Input shape: x={tuple(x.shape)}, w={tuple(w.shape)}")
    print()

    # --- Test 1: kernel with local scratch (expected to fail on hardware) ---
    print("=" * 60)
    print("TEST 1: kernel_with_local_scratch (nl.load_transpose2d from")
    print("        kernel-local nl.shared_hbm scratch)")
    print("Expected: NEFF compilation failure on trn1 hardware.")
    print("=" * 60)
    try:
        result1 = kernel_with_local_scratch(x, w)
        torch_xla.sync()
        print(f"RESULT: ran successfully, output shape: {tuple(result1.shape)}")
        print("  → Constraint does not exist on this SDK version.")
    except Exception as e:
        print(f"RESULT: failed with exception:")
        print(f"  {type(e).__name__}: {e}")
    print()

    # --- Test 2: working kernel (should succeed on hardware) ---
    print("=" * 60)
    print("TEST 2: kernel_with_external_arg (nl.load_transpose2d from")
    print("        function-argument HBM — known-good workaround)")
    print("Expected: succeeds on trn1 hardware.")
    print("=" * 60)
    try:
        result2 = kernel_with_external_arg(x, w)
        torch_xla.sync()
        print(f"RESULT: ran successfully, output shape: {tuple(result2.shape)}")
    except Exception as e:
        print(f"RESULT: UNEXPECTED failure:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)

    print()
    print("Reproducer complete.")
    print("Compiler artifacts (if --pipeline compile SaveTemps was set)")
    print("are in the neuronx-cc working directory listed in the log above.")
