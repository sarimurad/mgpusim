#!/usr/bin/env bash
# Attacker-view live re-key sweep. Set REPEAT=1 and smaller X/Y/Z for a smoke run.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

X="${X:-1024}" Y="${Y:-1024}" Z="${Z:-1024}"
ROUND_CYCLES="${ROUND_CYCLES:-177984}"
REPEAT="${REPEAT:-1}"
KREPEAT="${KREPEAT:-8}"
JOBS="${JOBS:-4}"
RESULTS="${RESULTS:-$here/../../../results/live-rekey-probe}"
RUNS="${RUNS:-$RESULTS/runs}"
OUT="${OUT:-$RESULTS/analysis}"
mkdir -p "$RUNS" "$OUT"

go build -o liverekeyprobe . || { echo "build failed"; exit 1; }
common=(-timing -x="$X" -y="$Y" -z="$Z" -repeat="$KREPEAT" -round-cycles="$ROUND_CYCLES" -seed=1234)
pids=()

run_one() {
  local name="$1"
  shift
  if [[ -s "$RUNS/${name}_summary.json" && -s "$RUNS/${name}_matrix.csv" && -s "$RUNS/${name}_epochs.csv" ]]; then
    echo "$(date '+%F %T') skip completed $name"
    return 0
  fi
  echo "$(date '+%F %T') start $name: $*"
  if ( time ./liverekeyprobe "${common[@]}" "$@" -out="$RUNS/$name" ) >"$RUNS/$name.log" 2>&1; then
    echo "$(date '+%F %T') done  $name: $(grep '>>> rounds' "$RUNS/$name.log" | tail -1)"
  else
    echo "$(date '+%F %T') FAILED $name (see $RUNS/$name.log); reported as missing"
    python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).unlink(missing_ok=True)' "$RUNS/${name}_summary.json"
    return 1
  fi
}

launch() {
  while (( ${#pids[@]} >= JOBS )); do
    wait "${pids[0]}" || true
    pids=("${pids[@]:1}")
  done
  run_one "$@" &
  pids+=("$!")
}

# One idle control, followed by REPEAT static and keyed runs per period.
launch idle_r00 -idle
for ((rep=1; rep<=REPEAT; rep++)); do
  suffix="$(printf '%02d' "$rep")"
  launch "static_r$suffix" -rekey-every=0
  for k in 1 2 4 8 16 32 64; do
    launch "k${k}_r$suffix" -rekey-every="$k"
  done
done
for pid in "${pids[@]}"; do wait "$pid" || true; done

PY="${PYTHON:-python3}"
[[ -x "$here/../../../.venv/bin/python" ]] && PY="${PYTHON:-$here/../../../.venv/bin/python}"
"$PY" analyze.py --runs-dir "$RUNS" --output-dir "$OUT"
