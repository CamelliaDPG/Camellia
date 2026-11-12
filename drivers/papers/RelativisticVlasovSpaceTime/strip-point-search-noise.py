#!/usr/bin/env python3
"""Remove MeshTopology's "parent matches, but none of its children do" diagnostics from a console log.

Each occurrence is that line, "parent vertices:", "child N, vertices:", and a FieldContainer dump for the
parent and each child.  The lines of every dump are recognised individually (not as a block), so output
from different MPI ranks that is interleaved with them is kept.  Everything else is copied unchanged.

usage: strip-point-search-noise.py console.log [cleaned.log]     (default output: console-clean.log beside the input)
"""
import os
import re
import sys

NOISE = [re.compile(p) for p in (
    r"^parent matches, but none of its children do\.\.\. will return nearest cell centroid\s*$",
    r"^parent vertices:\s*$",
    r"^child \d+, vertices:\s*$",
    r"^=+\s*$",                                           # dump borders
    r"^\s*Container (size|rank) = \d+\s*$",
    r"^\s*Dimensions\s*=(\s*\(\d+\))+\s*$",
    r"^\|\s*Multi-index\s+Enumeration\s+Value\s*\|\s*$",
    r"^\s+\d+(\s\d+)*\s+\d+\s+[-+]?\d\.\d+e[-+]\d+\s*$",  # dump rows: multi-index, enumeration, value
)]


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    source = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(source) or ".", "console-clean.log")
    occurrences = removed = kept = 0
    previous_removed = False
    with open(source, errors="replace") as fin, open(target, "w") as fout:
        for line in fin:
            if any(p.match(line) for p in NOISE):
                if line.startswith("parent matches"):
                    occurrences += 1
                removed += 1
                previous_removed = True
                continue
            if previous_removed and not line.strip():
                removed += 1  # the blank lines that separate the dumps
                continue
            previous_removed = False
            fout.write(line)
            kept += 1
    print(f"{source}: removed {occurrences} diagnostics ({removed} lines); kept {kept} lines in {target}")


if __name__ == "__main__":
    main()
