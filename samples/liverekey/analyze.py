#!/usr/bin/env python3
"""Analyze live-rekey runs.

Every number here comes from a run's output files. The matrices hold
hypervisor-side victim fills per set (not attacker probe contention), so each
metric is labelled as computed on fills. The only analytical values are the
Chapter 7 Table 7.4 bound, shown alongside for comparison.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Report Table 7.4: analytical bound raw / (R + 1), raw = 10 for 1024^3.
TABLE_7_4 = [(0, 10.0), (1, 5.0), (4, 2.0), (16, 0.6), (64, 0.2)]
SWEEP_KS = [1, 2, 4, 8, 16, 32]


def read_matrix(path):
    if path.stat().st_size == 0:
        return np.zeros((0, 256), dtype=np.int64)
    matrix = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def load_run(runs_dir, name):
    """Return (matrix, summary, epochs) or None if the run is missing."""
    summary_path = runs_dir / f"{name}_summary.json"
    if not summary_path.exists():
        return None
    return (
        read_matrix(runs_dir / f"{name}_matrix.csv"),
        json.loads(summary_path.read_text()),
        pd.read_csv(runs_dir / f"{name}_epochs.csv"),
    )


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")


def check_run(name, k, summary, epochs):
    """Acceptance checks that one run can answer on its own."""
    checks = {"accounting (row sums == victim fills)": bool(summary["accounting_match"])}
    if k > 0:
        expect = summary["active_rounds"] // k
        checks[f"rotations == floor(active/k) ({summary['rotations']} vs {expect})"] = (
            summary["rotations"] == expect
        )
        # A re-key is applied at the end of a full row and its flush count is
        # recorded on that row, so every rotation needs one row with flushes.
        rekey_rows = epochs[(epochs.full_round == 1) & (epochs.flushed_this_round > 0)]
        n_rekeys = summary["rotations"]
        checks[f"flushed > 0 at every re-key ({len(rekey_rows)}/{n_rekeys})"] = (
            len(rekey_rows) == n_rekeys
        )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    report_checks = []
    missing = []

    # Idle control.
    idle = load_run(args.runs_dir, "idle")
    if idle is None:
        missing.append("idle")
    else:
        m, s, e = idle
        report_checks.append(("idle", "matrix all zero", bool(m.size == 0 or not m.any())))
        np.savez_compressed(out / "idle.npz", trace=m.astype(np.int16),
                            key_epoch=e.key_epoch.to_numpy(dtype=np.int64))

    static = load_run(args.runs_dir, "static")
    if static is None:
        raise SystemExit("static run (k=0) is missing; nothing to compare against")
    s_matrix, s_sum, _ = static
    reference = s_matrix.sum(axis=0).astype(float)
    footprint0 = reference > 0
    ref_aligned = reference[footprint0].sum()

    runs = [("static", 0, static)]
    for k in SWEEP_KS:
        run = load_run(args.runs_dir, f"k{k}")
        if run is None:
            missing.append(f"k{k}")
        else:
            runs.append((f"k{k}", k, run))

    rows = []
    for name, k, (matrix, summary, epochs) in runs:
        cols = matrix.sum(axis=0).astype(float)
        walks0 = s_sum["victim_walks"]
        t0 = s_sum["sim_time_s"]
        rows.append({
            "rekey_every": k,
            "rounds": summary["rounds"],
            "active_rounds": summary["active_rounds"],
            "rotations": summary["rotations"],
            "raw_total_fills": summary["raw_total"],
            "aligned_recovery_fills": cols[footprint0].sum() / ref_aligned if ref_aligned else float("nan"),
            "fingerprint_cosine": cosine(cols, reference),
            "nonzero_sets": int(np.count_nonzero(cols)),
            "victim_walks": summary["victim_walks"],
            "extra_walks_pct": 100.0 * (summary["victim_walks"] - walks0) / walks0 if walks0 else float("nan"),
            "sim_time_s": summary["sim_time_s"],
            "sim_time_change_pct": 100.0 * (summary["sim_time_s"] - t0) / t0 if t0 else float("nan"),
            "flushed_entries": summary["flushed_entries"],
            "L2_crossPID": summary["L2_crossPID"],
        })
        for label, ok in check_run(name, k, summary, epochs).items():
            report_checks.append((name, label, ok))
        np.savez_compressed(out / f"{name}.npz", trace=matrix.astype(np.int16),
                            key_epoch=epochs.key_epoch.to_numpy(dtype=np.int64))

    frame = pd.DataFrame(rows).sort_values("rekey_every")
    frame.to_csv(out / "liverekey_results.csv", index=False)

    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)  # noqa: E731
    cols = ["rekey_every", "rotations", "raw_total_fills", "aligned_recovery_fills",
            "fingerprint_cosine", "extra_walks_pct", "sim_time_change_pct", "flushed_entries"]
    md = [
        "# Live re-key results",
        "",
        "**Measured.** Every value in the first table comes from a simulator run",
        "(victim 1024³ matrix multiply unless the run logs say otherwise).",
        "The matrices are hypervisor-side victim fills per L2 TLB set, not attacker",
        "probe contention, so `aligned_recovery_fills` is the share of the victim's fills",
        "that still land in the static-run footprint F0. It is not the same quantity as",
        "the \"Recoverable\" column of Table 7.4.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in frame.to_dict("records"):  # keeps int columns as ints
        md.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    md += [
        "",
        "**Analytical (Chapter 7, Table 7.4)**: raw / (R + 1) with raw = 10. Not measured.",
        "",
        "| R (rekeys) | Recoverable (analytical) |",
        "|---|---|",
    ]
    md += [f"| {r} | {v} |" for r, v in TABLE_7_4]
    md += ["", "## Acceptance checks (from run outputs)", "",
           "| Run | Check | Result |", "|---|---|---|"]
    md += [f"| {n} | {c} | {'PASS' if ok else 'FAIL'} |" for n, c, ok in report_checks]
    if missing:
        md += ["", "**Missing runs:** " + ", ".join(missing) + " (failed or not run)."]
    (out / "liverekey_results.md").write_text("\n".join(md) + "\n")

    keyed = frame[frame.rekey_every > 0]
    fig, left = plt.subplots(figsize=(8, 4.8))
    if not keyed.empty:
        left.plot(keyed.rekey_every, keyed.aligned_recovery_fills, "o-",
                  label="Aligned recovery (fills in F0, vs static)")
        left.plot(keyed.rekey_every, keyed.fingerprint_cosine, "^-",
                  label="Fingerprint cosine vs static")
        right = left.twinx()
        right.plot(keyed.rekey_every, keyed.extra_walks_pct, "s--", color="tab:red",
                   label="Extra victim walks (%)")
        right.set_ylabel("Extra victim walks vs static (%)")
        h1, l1 = left.get_legend_handles_labels()
        h2, l2 = right.get_legend_handles_labels()
        left.legend(h1 + h2, l1 + l2, loc="best")
    left.set_xscale("log", base=2)
    left.set_xlabel("Re-key period k (rounds)")
    left.set_ylabel("Fraction of static run")
    top = max([1.0] + keyed.aligned_recovery_fills.dropna().tolist())
    left.set_ylim(0, top * 1.08)
    left.set_title("Live re-keying: measured footprint vs overhead")
    fig.tight_layout()
    fig.savefig(out / "liverekey_tradeoff.png", dpi=160)

    print(frame.to_string(index=False))
    failed = [c for c in report_checks if not c[2]]
    print(f"checks: {len(report_checks) - len(failed)} pass, {len(failed)} fail; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
