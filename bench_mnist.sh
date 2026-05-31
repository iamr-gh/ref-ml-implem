#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIM_OUT="$ROOT/nim/mnist/nim_profile.txt"
ODIN_OUT="$ROOT/odin/mnist/odin_profile.txt"

if [[ $# -eq 0 ]]; then
  ARGS=(none)
else
  ARGS=("$@")
fi

pushd "$ROOT/nim/mnist" >/dev/null
nim c -d:release main.nim >/dev/null
./main "${ARGS[@]}" | tee "$NIM_OUT"
popd >/dev/null

pushd "$ROOT" >/dev/null
odin build odin/mnist -o:speed -out:odin/mnist/mnist >/dev/null
popd >/dev/null

pushd "$ROOT/odin/mnist" >/dev/null
./mnist "${ARGS[@]}" | tee "$ODIN_OUT"
popd >/dev/null

python3 "$ROOT/compare_profiles.py" "$NIM_OUT" "$ODIN_OUT"