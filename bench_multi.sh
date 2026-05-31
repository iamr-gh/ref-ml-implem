#!/usr/bin/env bash
# Multi-run benchmark for Nim vs Odin MNIST.
# Usage: ./bench_multi.sh [NUM_RUNS] [HIDDEN]
# Defaults: 3 runs, hidden=128
set -euo pipefail

NUM_RUNS="${1:-3}"
HIDDEN="${2:-128}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NIM_BIN="$SCRIPT_DIR/nim/mnist/main"
ODIN_BIN="$SCRIPT_DIR/odin/mnist/main"
RESULTS_DIR="/tmp/mnist_bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== MNIST Benchmark: $NUM_RUNS runs, hidden=$HIDDEN ==="
echo "Results: $RESULTS_DIR"

# Build
echo "Building..."
cd "$SCRIPT_DIR/nim/mnist" && nim c -d:release --threads:off main.nim 2>&1 | tail -1
cd "$SCRIPT_DIR/odin/mnist" && odin build main.odin -file -o:speed -no-bounds-check 2>&1 | tail -1

echo ""
for i in $(seq 1 "$NUM_RUNS"); do
    echo "Nim run $i/$NUM_RUNS..."
    cd "$SCRIPT_DIR/nim/mnist" && "$NIM_BIN" $HIDDEN > "$RESULTS_DIR/nim_run${i}.txt" 2>&1
done
for i in $(seq 1 "$NUM_RUNS"); do
    echo "Odin run $i/$NUM_RUNS..."
    cd "$SCRIPT_DIR/odin/mnist" && "$ODIN_BIN" $HIDDEN > "$RESULTS_DIR/odin_run${i}.txt" 2>&1
done

echo ""
echo "=== Results ==="

# Parse with Python
python3 - "$RESULTS_DIR" "$NUM_RUNS" <<'PYEOF'
import sys, os, re, statistics

results_dir = sys.argv[1]
num_runs = int(sys.argv[2])

def parse_timing(path):
    timings = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'TIMING\s+(\S+)\s+(\S+)s', line)
            if m:
                timings[m.group(1)] = float(m.group(2))
    return timings

def avg_epochs(timings, prefix):
    vals = [v for k, v in timings.items() if re.match(rf'epoch_\d+_{prefix}', k)]
    return statistics.mean(vals) if vals else None

nim_runs = [parse_timing(os.path.join(results_dir, f"nim_run{i}.txt")) for i in range(1, num_runs+1)]
odin_runs = [parse_timing(os.path.join(results_dir, f"odin_run{i}.txt")) for i in range(1, num_runs+1)]

def stats(vals):
    if not vals or len(vals) == 0:
        return None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0
    cv = (s / m * 100) if m > 0 else 0
    return m, s, cv

print(f"\n{'Metric':<22} {'Nim mean':>10} {'±std':>8} {'CV%':>6} │ {'Odin mean':>10} {'±std':>8} {'CV%':>6} │ {'ratio':>7}")
print("─" * 90)

metrics = [
    ("load_train", "Load train"),
    ("init_model", "Init model"),
    ("baseline_eval", "Baseline eval"),
]

for label, display in metrics:
    nv = [r.get(label) for r in nim_runs if label in r]
    ov = [r.get(label) for r in odin_runs if label in r]
    ns = stats(nv); os_ = stats(ov)
    nm, nsd, ncv = ns if ns else (0,0,0)
    om, osd, ocv = os_ if os_ else (0,0,0)
    ratio = nm/om if om > 0 else float('inf')
    print(f"{display:<22} {nm:10.4f} {nsd:8.4f} {ncv:5.1f}% │ {om:10.4f} {osd:8.4f} {ocv:5.1f}% │ {ratio:7.3f}")

# Epoch averages
def derived_row(display, nim_data, odin_data):
    if nim_data is None or odin_data is None:
        print(f"{display:<22} {'N/A':>10}")
        return
    nm, nsd, ncv = nim_data
    om, osd, ocv = odin_data
    ratio = nm/om if om > 0 else float('inf')
    print(f"{display:<22} {nm:10.4f} {nsd:8.4f} {ncv:5.1f}% │ {om:10.4f} {osd:8.4f} {ocv:5.1f}% │ {ratio:7.3f}")

nim_trains = [avg_epochs(r, "train") for r in nim_runs]
odin_trains = [avg_epochs(r, "train") for r in odin_runs]
nim_evals = [avg_epochs(r, "eval") for r in nim_runs]
odin_evals = [avg_epochs(r, "eval") for r in odin_runs]

derived_row("Avg epoch train", stats(nim_trains), stats(odin_trains))
derived_row("Avg epoch eval", stats(nim_evals), stats(odin_evals))

nt = [r.get("total") for r in nim_runs if "total" in r]
ot = [r.get("total") for r in odin_runs if "total" in r]
derived_row("Total", stats(nt), stats(ot))

# Per-run detail
print(f"\n--- Per-run train epoch averages ---")
for i, r in enumerate(nim_runs):
    t = avg_epochs(r, "train")
    e = avg_epochs(r, "eval")
    print(f"  Nim  run {i+1}: train={t:.4f}s  eval={e:.4f}s  total={r.get('total',0):.2f}s")
for i, r in enumerate(odin_runs):
    t = avg_epochs(r, "train")
    e = avg_epochs(r, "eval")
    print(f"  Odin run {i+1}: train={t:.4f}s  eval={e:.4f}s  total={r.get('total',0):.2f}s")

# Final accuracy
print(f"\n--- Final test accuracy ---")
for i, r in enumerate(nim_runs):
    path = os.path.join(results_dir, f"nim_run{i+1}.txt")
    last = ""
    with open(path) as f:
        for line in f:
            if "test acc=" in line and "Epoch" in line:
                last = line.strip()
    # extract accuracy
    m = re.search(r'test acc=\s*(\S+)%', last)
    acc = m.group(1) if m else "?"
    print(f"  Nim  run {i+1}: {acc}%")
for i, r in enumerate(odin_runs):
    path = os.path.join(results_dir, f"odin_run{i+1}.txt")
    last = ""
    with open(path) as f:
        for line in f:
            if "test acc=" in line and "Epoch" in line:
                last = line.strip()
    m = re.search(r'test acc=\s*(\S+)%', last)
    acc = m.group(1) if m else "?"
    print(f"  Odin run {i+1}: {acc}%")
PYEOF