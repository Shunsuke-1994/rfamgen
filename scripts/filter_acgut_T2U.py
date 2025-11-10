#!/usr/bin/env python3
"""
Filter FASTA sequences to keep only those composed of A/C/G/U/T (case-insensitive),
remove gap characters '-', convert T to U, and write the cleaned sequences.

This script is a minimal extraction of the logic:
 - Exclude sequences containing any character other than A/C/G/U/T
 - Convert T -> U in output

Usage:
  python scripts/filter_acgut_T2U.py --input INPUT.fa --output OUTPUT.fa

If --output is omitted, an automatic filename will be generated
from the input path by appending "_filtered_T2U.fa".
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator, Tuple


ALLOWED = set("ACGUT")


def parse_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Simple FASTA parser yielding (header, sequence) pairs.

    - header: the line without the leading '>' (stripped of trailing newline)
    - sequence: the concatenated sequence lines as-is (no spaces/newlines)
    """
    header = None
    seq_lines = []
    open_fn = open
    if path == "-":  # read from stdin
        fh = sys.stdin
    else:
        fh = open_fn(path, "r")
    try:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            yield header, "".join(seq_lines)
    finally:
        if fh is not sys.stdin:
            fh.close()


def clean_and_filter(seq: str) -> str | None:
    """Uppercase, remove gaps '-', verify only A/C/G/U/T, then convert T->U.

    Returns cleaned sequence if valid; otherwise None.
    """
    s = seq.upper().replace("-", "").replace(" ", "")
    if not s:
        return None
    if set(s) <= ALLOWED:
        return s.replace("T", "U")
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Filter FASTA to A/C/G/U/T and convert T->U")
    p.add_argument("--input", "-i", required=True, help="Input FASTA file (use '-' for stdin)")
    p.add_argument("--output", "-o", help="Output FASTA path (default: <input>_filtered_T2U.fa)")
    args = p.parse_args(argv)

    in_path = args.input
    if args.output:
        out_path = args.output
    else:
        base = "stdin" if in_path == "-" else os.path.splitext(in_path)[0]
        out_path = f"{base}_filtered_T2U.fa"

    kept = 0
    dropped = 0
    with (sys.stdout if out_path == "-" else open(out_path, "w")) as out_fh:
        for header, seq in parse_fasta(in_path):
            cleaned = clean_and_filter(seq)
            if cleaned is None:
                dropped += 1
                continue
            kept += 1
            out_fh.write(f">{header}\n")
            out_fh.write(f"{cleaned}\n")

    # Progress summary to stderr to avoid polluting FASTA output when writing to stdout
    print(f"kept={kept} dropped={dropped} output={out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
