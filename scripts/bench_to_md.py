#!/usr/bin/env python3
"""Convert pytest-benchmark JSON to a markdown table for docs/benchmarks.md.

Usage:
    python scripts/bench_to_md.py benchmark_results.json > table.md
    python scripts/bench_to_md.py benchmark_results.json --inplace docs/benchmarks.md

The --inplace mode replaces the content between the marker lines:
    <!-- BENCH_TABLE_START -->
    ... (replaced) ...
    <!-- BENCH_TABLE_END -->
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Match the pytest-benchmark test names produced by bench_fft.py:
#   benchmarks/bench_fft.py::TestFFT1D::test_fft_nki[8]
#   benchmarks/bench_fft.py::TestFFT2D::test_fft2_torch[fft2_shape0]
NAME_RE = re.compile(
    r"::Test(?P<group>\w+)::test_(?P<op>\w+?)_"
    r"(?P<variant>nki|trnfft_pytorch|torch(?:_complex64)?)"
    r"(?:\[(?P<param>.+)\])?$"
)

VARIANT_LABEL = {
    "nki": "NKI",
    "trnfft_pytorch": "trnfft-PyTorch",
    "torch": "torch.*",
    "torch_complex64": "torch.* (c64)",
}


def parse_results(path: Path) -> dict:
    """Return {(group, op, param): {variant: median_us}}."""
    data = json.loads(path.read_text())
    rows: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "") + (f"[{bench['param']}]" if bench.get("param") else "")
        full = bench.get("fullname", name)
        m = NAME_RE.search(full)
        if not m:
            print(f"warn: cannot parse {full}", file=sys.stderr)
            continue
        group = m.group("group")
        op = m.group("op")
        variant = m.group("variant")
        param = m.group("param") or ""
        median_s = bench["stats"]["median"]
        rows[(group, op, param)][variant] = median_s * 1e6  # to μs
    return rows


def render_markdown(rows: dict, machine_info: dict | None = None) -> str:
    out = []
    if machine_info:
        host = machine_info.get("node", "")
        cpu = machine_info.get("cpu", {}).get("brand_raw", "")
        out.append(f"_Hardware: {host} — {cpu}_\n")

    out.append("| Operation | Variant | Param | Median (μs) | vs trnfft-PyTorch |")
    out.append("|-----------|---------|-------|------------:|------------------:|")

    for (group, op, param) in sorted(rows.keys()):
        variants = rows[(group, op, param)]
        baseline = variants.get("trnfft_pytorch")
        for variant in ("nki", "trnfft_pytorch", "torch", "torch_complex64"):
            if variant not in variants:
                continue
            us = variants[variant]
            label = VARIANT_LABEL[variant]
            if variant == "trnfft_pytorch" or baseline is None:
                speedup = ""
            else:
                ratio = baseline / us
                if ratio >= 1.0:
                    speedup = f"{ratio:.2f}× faster"
                else:
                    speedup = f"{1/ratio:.2f}× slower"
            param_disp = param.replace("_", " ") if param else "-"
            out.append(
                f"| {group}.{op} | {label} | {param_disp} | {us:>10.2f} | {speedup} |"
            )
    return "\n".join(out) + "\n"


def replace_inplace(doc_path: Path, table_md: str) -> None:
    text = doc_path.read_text()
    pattern = re.compile(
        r"(<!-- BENCH_TABLE_START -->)(.*?)(<!-- BENCH_TABLE_END -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        print(f"error: no marker block in {doc_path}", file=sys.stderr)
        sys.exit(1)
    new = pattern.sub(rf"\1\n\n{table_md}\n\3", text)
    doc_path.write_text(new)
    print(f"updated {doc_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", type=Path)
    ap.add_argument("--inplace", type=Path, help="replace marker block in this doc")
    args = ap.parse_args()

    if not args.results_json.exists():
        print(f"error: {args.results_json} not found", file=sys.stderr)
        return 1

    data = json.loads(args.results_json.read_text())
    rows = parse_results(args.results_json)
    md = render_markdown(rows, data.get("machine_info"))

    if args.inplace:
        replace_inplace(args.inplace, md)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
