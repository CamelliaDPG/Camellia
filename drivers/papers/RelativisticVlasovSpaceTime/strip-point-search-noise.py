#!/usr/bin/env python3
"""Remove MeshTopology's "parent matches, but none of its children do" diagnostics from a console log.

Each occurrence is that line, "parent vertices:", "child N, vertices:", and a FieldContainer dump for the
parent and each child.  When many ranks print these at once, their output is interleaved within lines,
so the script works on fragments: it deletes every recognisable piece of a diagnostic wherever it occurs,
keeps whatever else is on such a line, and drops lines left holding only debris (digits, rules, or pieces
of the diagnostic's words).  Lines without any of the diagnostic's words are kept unchanged unless they
hold nothing but numbers and rules.  A normal line glued to debris therefore survives, though possibly garbled.

usage: strip-point-search-noise.py console.log [cleaned.log]     (default output: console-clean.log beside the input)
"""
import os
import re
import sys

PHRASES = [
    "parent matches, but none of its children do... will return nearest cell centroid",
    "parent vertices:",
    "vertices:",
    "Container size =",
    "Container rank =",
    "Dimensions",
    "Multi-index",
    "Enumeration",
    "Value",
]
# pieces of a diagnostic: its phrases, "child N,", rules, dimension lists, and the numbers in dump rows
PIECES = re.compile("|".join([re.escape(p) for p in PHRASES] + [
    r"child \d+,",
    r"={3,}",
    r"^\t Container",
    r"\(\d+\)",
    r"[-+]?\d\.\d+e[-+]\d+",   # values
]))
# a line holding any of these carries part of a diagnostic (numbers alone do not say so: they are everywhere)
MARKERS = re.compile("|".join([re.escape(p) for p in PHRASES] + [r"child \d+,", r"={3,}", r"^\t Container"]))
DEBRIS = re.compile(r"^[\s\d.,:;|=()+\-e]*$")  # what is left of a dump line once its pieces are removed
NOISE_TEXT = "".join(PHRASES) + "childvertices:"


def is_debris(text):
    if DEBRIS.match(text):
        return True
    compact = re.sub(r"[\s\d]", "", text)
    if len(re.sub(r"[^A-Za-z]", "", compact)) <= 12:
        return True  # a few letters among the debris: pieces of words cut apart by other ranks' output
    # a fragment of the diagnostic's words, cut where another rank's output interrupted it
    return len(compact) <= 40 and compact in re.sub(r"\s", "", NOISE_TEXT)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    source = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(source) or ".", "console-clean.log")
    occurrences = removed = kept = salvaged = 0
    previous_removed = False
    with open(source, errors="replace") as fin, open(target, "w") as fout:
        for line in fin:
            occurrences += line.count("parent matches")
            if not MARKERS.search(line):
                if re.search(r"\d", line) and DEBRIS.match(line):
                    removed += 1  # a stray piece of a dump row
                    previous_removed = True
                    continue
                if previous_removed and not line.strip():
                    removed += 1  # blank lines that separate the dumps
                    continue
                previous_removed = False
                fout.write(line)
                kept += 1
                continue
            residual = PIECES.sub(" ", line.rstrip("\n"))
            if is_debris(residual):
                removed += 1
                previous_removed = True
                continue
            # keep the rest of the line, tidied
            fout.write(re.sub(r"\s{3,}", "  ", residual).strip() + "\n")
            salvaged += 1
            previous_removed = False
    print(f"{source}: removed {occurrences} diagnostics ({removed} lines); kept {kept} lines unchanged and "
          f"{salvaged} lines with diagnostic fragments removed, in {target}")


if __name__ == "__main__":
    main()
