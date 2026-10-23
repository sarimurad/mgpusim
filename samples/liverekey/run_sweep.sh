#!/usr/bin/env bash
# Runs the live re-key sweep, then analyze.py. A full 1024^3 sweep takes
# ~2 h, so start it in the background and watch runs/*.log:
#
#   nohup ./run_sweep.sh > runs/sweep.log 2>&1 &
#
# X, Y, Z and ROUND_CYCLES override the victim size and round length (for a
# quick smoke test, e.g. X=256 Y=256 Z=256 ROUND_CYCLES=20000 RUNS=smoke).
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

X="${X:-1024}" Y="${Y:-1024}" Z="${Z:-1024}"
ROUND_CYCLES="${ROUND_CYCLES:-177984}"
RUNS="${RUNS:-runs}"
mkdir -p "$RUNS"

go build -o liverekey . || { echo "build failed"; exit 1; }

common=(-timing -x="$X" -y="$Y" -z="$Z" -round-cycles="$ROUND_CYCLES" -seed=1234)

run_one() {
  local name="$1"
  shift
  echo "$(date '+%F %T') start $name: $*"
  if ( time ./liverekey "${common[@]}" "$@" -out="$RUNS/$name" ) >"$RUNS/$name.log" 2>&1; then
    echo "$(date '+%F %T') done  $name: $(grep '>>> rounds' "$RUNS/$name.log")"
  else
    echo "$(date '+%F %T') FAILED $name (see $RUNS/$name.log); reported as missing"
    rm -f "$RUNS/$name"_summary.json
  fi
}

run_one idle -idle
run_one static -rekey-every=0
for k in 1 2 4 8 16 32; do
  run_one "k$k" -rekey-every="$k"
done

# Uses ~/ENCS5300/.venv (numpy, pandas, matplotlib) when present.
PY="${PYTHON:-python3}"
[ -x "$here/../../../.venv/bin/python" ] && PY="${PYTHON:-$here/../../../.venv/bin/python}"
"$PY" analyze.py --runs-dir "$RUNS" --output-dir "$RUNS"
