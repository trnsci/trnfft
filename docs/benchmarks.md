# Benchmarks

trnfft NKI kernels measured against two baselines on the same Trainium instance:

1. **NKI** — trnfft with `set_backend("nki")` running on Tensor + Vector Engines
2. **trnfft-PyTorch** — trnfft with `set_backend("pytorch")` running on the host CPU
3. **torch.\*** — vanilla `torch.fft.*` / `torch.matmul` on the host CPU

The first comparison (1 vs 2) answers *"did our NKI kernels actually help vs our own PyTorch fallback?"* It uses the same code path on both sides — only the backend dispatch differs.

The second comparison (1 vs 3) answers *"what's the user-visible difference between trnfft and vanilla PyTorch?"*

## Methodology

- **Hardware**: AWS `trn1.2xlarge` (1 NeuronCore-v2, 32 GB SBUF, AMD EPYC host CPU)
- **Image**: Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)
- **Neuron SDK**: `neuronxcc 2.24.5133.0`
- **Tool**: [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) with default settings (calibration + 5+ rounds, reports median)
- **NKI warmup**: each NKI test does an explicit warmup call before the timed loop so kernel compilation cost is excluded from steady-state numbers
- **Reproduce**: `AWS_PROFILE=aws ./scripts/run_benchmarks.sh`

## Caveats

- **Small sizes are dispatch-bound.** NKI kernel invocation has fixed overhead (XLA dispatch + tensor staging). Below some per-op threshold the host CPU wins. We report this honestly — the goal is for users to see *where* NKI helps, not to cherry-pick favorable shapes.
- **Single-NeuronCore only.** trn1.2xlarge has one NeuronCore. Multi-NeuronCore parallelism (planned for v0.8) will change the picture for large transforms.
- **FP32 throughout.** BF16 / FP16 paths are future work.
- Numbers can vary 5-15% run-to-run; treat the table as approximate.

## Results

<!-- BENCH_TABLE_START -->

_Run `scripts/run_benchmarks.sh` and `scripts/bench_to_md.py` to populate this table._

<!-- BENCH_TABLE_END -->

## Raw data

Versioned JSON outputs are committed under `docs/benchmark_results/`. To compare across versions:

```bash
pytest-benchmark compare docs/benchmark_results/*.json
```
