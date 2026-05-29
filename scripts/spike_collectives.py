"""Spike: is nki.collectives.all_reduce available and callable?

Corrected against the real NKI 0.4.0 API (docs verified 2026-05-28):
    nki.collectives.all_reduce(srcs: List, dsts: List, replica_group: ReplicaGroup,
                               op, priority=None)
    op in {nl.add, nl.minimum, nl.maximum}

The SDK230_SPIKE_PROMPTS.md guessed `allreduce(partial, result, op=...SUM)` — wrong on
name, arg shape, and op constant. This script uses the documented signature.

Key question for trnfft: collectives were introduced in NKI 0.2.0 (not 2.30), so they
should be present on the SDK 2.29 AMI. This confirms whether the NeuronLink stage-parallel
FFT path was ever actually gated.
"""

import inspect
import sys


def main():
    import neuronxcc

    print(f"neuronxcc version: {neuronxcc.__version__}")

    try:
        import nki

        print(f"nki version: {getattr(nki, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"nki import FAILED: {e}")
        sys.exit(1)

    # Check 1: is nki.collectives importable?
    print("\n[1] nki.collectives importable:", end=" ")
    try:
        import nki.collectives as ncc

        print("YES")
        names = [x for x in dir(ncc) if not x.startswith("_")]
        print(f"    public names: {names}")
    except ImportError as e:
        print(f"NO: {e}")
        sys.exit(0)

    # Check 2: all_reduce signature
    print("\n[2] all_reduce signature:", end=" ")
    try:
        print(inspect.signature(ncc.all_reduce))
    except Exception as e:
        print(f"introspect failed: {e}")

    # Check 3: ReplicaGroup available?
    print("[3] ReplicaGroup available:", end=" ")
    try:
        print(f"YES — {ncc.ReplicaGroup}")
        print(f"    ReplicaGroup signature: {inspect.signature(ncc.ReplicaGroup)}")
    except Exception as e:
        print(f"NO / introspect failed: {e}")

    # Check 4: which collective ops exist?
    print("[4] available collectives:")
    for name in [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "all_to_all_v",
        "collective_permute",
        "rank_id",
    ]:
        print(f"    {name}: {'YES' if hasattr(ncc, name) else 'no'}")

    # Check 5: minimal single-core all_reduce kernel (replica group of 1 — does it
    # at least compile and run, confirming the calling convention?)
    print("\n[5] minimal all_reduce kernel (1-rank replica group):", end=" ")
    try:
        import torch

        import nki.language as nl

        @nki.jit
        def _allreduce_test(x):
            src = nl.load(x[0:128, 0:128])
            dst = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
            rg = ncc.ReplicaGroup([[0]])  # single rank
            ncc.all_reduce([src], [dst], replica_group=rg, op=nl.add)
            return dst

        xt = torch.ones(128, 128)
        r = _allreduce_test(xt)
        val = r[0, 0].item()
        print(f"OK — result[0,0]={val} (expected ~1.0 for 1-rank)")
    except Exception as e:
        print("FAILED")
        print(f"    {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
