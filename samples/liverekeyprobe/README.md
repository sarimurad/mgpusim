# Live re-key attacker-view probe

`main.go` models a translation-level attacker against the shared L2 TLB used by
the matrix-multiplication victim. It uses the same `GPUs[0].L2TLBs[0]` that the
existing `primeprobe` sample uses. The attacker PID defaults to 9000; it does
not run a GPU kernel. Its eviction-set addresses are inserted directly into
the TLB model before `r.Run()`, so this experiment measures TLB residency and
does not model attacker instruction execution, wall-clock timing, or a physical
GPU side channel.

The primary `<prefix>_matrix.csv` is the attacker probe matrix, with 256 values
per row and values from 0 through the TLB associativity. The sample also writes
`<prefix>_victim_fills.csv` as a separate hypervisor-side view,
`<prefix>_epochs.csv` with per-round accounting, and `<prefix>_summary.json`.
`analyze.py` validates the runs and emits one `.npz` per run with `trace`
(`int16`, `[R,256]`) and `key_epoch` (`[R]`).

Build and run a short smoke sweep:

```bash
REPEAT=1 JOBS=2 X=128 Y=128 Z=128 ROUND_CYCLES=10000 \
  RUNS=/tmp/probe-runs OUT=/tmp/probe-analysis ./run_sweep.sh
```

The default sweep uses a 1024³ victim and eight repeats. Start it in the
background and monitor the per-run logs:

```bash
nohup ./run_sweep.sh > sweep.log 2>&1 &
```

Use `JOBS` to limit concurrent simulations. The default is two because each
timing simulation is CPU and memory intensive.
