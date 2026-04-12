#!/usr/bin/env python3
"""Parse pytest-benchmark TEXT output (when JSON wasn't captured) into the
markdown table for docs/benchmarks.md.

Why this exists: SSM stdout is capped at 24KB and our base64'd JSON exceeded
that limit on the first run. The pretty-printed text table from pytest-benchmark
has all the same medians we need.

Usage:
    python scripts/bench_text_to_md.py path/to/captured-output.txt --inplace docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Lines look like:
#   test_fft_torch[256]   8.7780 (1.0)   29.7400 (1.0)   13.1522 (1.0)   9.2787 (6.10)   8.8680 (1.0)   5.8202 (49.96)   1;1   76,032.9069 (1.0)   5   1
# Median is the 5th numeric column (after stripping the (xx) ratios).
LINE_RE = re.compile(r"^(test_\S+)\s+(.*)$")
NUM_RE = re.compile(r"([\d,]+\.\d+)\s*\([>\d.]+\)")
NAME_RE = re.compile(
    r"^test_(?P<op>\w+?)_(?P<variant>nki|trnfft_pytorch|torch(?:_complex64)?)"
    r"(?:\[(?P<param>.+)\])?$"
)

VARIANT_LABEL = {
    "nki": "NKI",
    "trnfft_pytorch": "trnfft-PyTorch",
    "torch": "torch.*",
    "torch_complex64": "torch.* (c64)",
}


def parse_line(line: str) -> tuple[str, float] | None:
    m = LINE_RE.match(line)
    if not m:
        return None
    name = m.group(1)
    rest = m.group(2)
    nums = NUM_RE.findall(rest)
    if len(nums) < 5:
        return None
    median_str = nums[4].replace(",", "")
    return name, float(median_str)


def parse_file(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(op, param): {variant: median_us}}."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for raw in path.read_text().splitlines():
        parsed = parse_line(raw)
        if parsed is None:
            continue
        name, median_us = parsed
        m = NAME_RE.match(name)
        if not m:
            continue
        op = m.group("op")
        variant = m.group("variant")
        param = m.group("param") or ""
        out[(op, param)][variant] = median_us
    return out


def render_markdown(rows: dict) -> str:
    out = [
        "_Hardware: AWS trn1.2xlarge — neuronxcc 2.24.5133.0 — Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)_",
        "",
        "_All times in microseconds (μs); lower is better. Speedup is trnfft-PyTorch / NKI._",
        "",
        "| Operation | Param | NKI | trnfft-PyTorch | torch.* | NKI vs PyT |",
        "|-----------|-------|----:|---------------:|--------:|----------:|",
    ]
    for (op, param) in sorted(rows.keys(), key=lambda k: (k[0], k[1])):
        v = rows[(op, param)]
        nki = v.get("nki")
        pyt = v.get("trnfft_pytorch")
        torch_v = v.get("torch") or v.get("torch_complex64")
        param_disp = param if param else "-"
        nki_str = f"{nki:,.1f}" if nki is not None else "—"
        pyt_str = f"{pyt:,.1f}" if pyt is not None else "—"
        torch_str = f"{torch_v:,.1f}" if torch_v is not None else "—"
        if nki is not None and pyt is not None:
            ratio = pyt / nki
            speedup = f"{ratio:.2f}× faster" if ratio >= 1.0 else f"{1/ratio:.2f}× slower"
        else:
            speedup = "—"
        out.append(f"| {op} | {param_disp} | {nki_str} | {pyt_str} | {torch_str} | {speedup} |")
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
    ap.add_argument("text_output", type=Path)
    ap.add_argument("--inplace", type=Path)
    args = ap.parse_args()
    rows = parse_file(args.text_output)
    md = render_markdown(rows)
    if args.inplace:
        replace_inplace(args.inplace, md)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
