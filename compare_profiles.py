#!/usr/bin/env python3
"""Compare TIMING lines from Nim and Odin MNIST profile outputs.

Usage:
    python3 compare_profiles.py nim_profile.txt odin_profile.txt
"""

import re
import sys

TIMING_RE = re.compile(r"^TIMING\s+(\S+)\s+([0-9.]+)s")


def read_timings(path):
    timings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = TIMING_RE.match(line)
            if m:
                timings[m.group(1)] = float(m.group(2))
    return timings


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)

    nim = read_timings(sys.argv[1])
    odin = read_timings(sys.argv[2])
    keys = sorted(set(nim) | set(odin))

    print("Phase                 Nim(s)    Odin(s)   Odin/Nim")
    print("-------------------  --------  --------  --------")
    for k in keys:
        n = nim.get(k)
        o = odin.get(k)
        if n is None or o is None:
            print(f"{k:<19}  {n or '-':>8}  {o or '-':>8}  {'-':>8}")
        else:
            print(f"{k:<19}  {n:8.4f}  {o:8.4f}  {o / n:8.3f}")


if __name__ == "__main__":
    main()