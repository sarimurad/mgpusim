#!/usr/bin/env python3
"""Validate and export attacker-view live-rekey runs."""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^(idle|static|k\d+)_r\d+$")
KS = [1, 2, 4, 8, 16, 32, 64]


def load_run(prefix):
    matrix_path = prefix.with_name(prefix.name + "_matrix.csv")
    epochs_path = prefix.with_name(prefix.name + "_epochs.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    if not summary_path.exists():
        return None
    matrix = np.loadtxt(matrix_path, delimiter=",", dtype=np.int64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    epochs = pd.read_csv(epochs_path)
    summary = json.loads(summary_path.read_text())
    return matrix, epochs, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    matrices = {}
    epoch_tables = {}
    summaries = {}
    failures = []
    for summary_path in sorted(args.runs_dir.glob("*_summary.json")):
        prefix = summary_path.with_name(summary_path.name.removesuffix("_summary.json"))
        name = prefix.name
        if not RUN_RE.match(name):
            continue
        loaded = load_run(prefix)
        if loaded is None:
            failures.append(f"{name}: missing summary")
            continue
        matrix, epochs, summary = loaded
        matrices[name] = matrix
        epoch_tables[name] = epochs
        summaries[name] = summary
        checks = {
            "shape_256": matrix.shape[1] == 256,
            "values_0_to_8": bool(matrix.size == 0 or (matrix.min() >= 0 and matrix.max() <= 8)),
            "epoch_rows_match": len(epochs) == matrix.shape[0],
            "victim_fill_accounting": bool(summary.get("accounting_match", False)),
        }
        if name.startswith("idle_"):
            checks["idle_zero"] = bool(matrix.size == 0 or not matrix.any())
        if name.startswith("k"):
            k = int(name[1:].split("_", 1)[0])
            expected_rotations = int(summary["active_rounds"]) // k
            checks["rotation_count"] = int(summary["rotations"]) == expected_rotations
            rotated = epochs[(epochs.full_round == 1) & (epochs.flushed_this_round > 0)]
            checks["flush_each_rotation"] = len(rotated) == int(summary["rotations"])
        if name.startswith("static_"):
            checks["static_cross_pid_present"] = int(summary["L2_crossPID"]) > 0
            checks["static_probe_nonzero"] = bool(matrix.any())
        if not all(checks.values()):
            failures.append(f"{name}: " + ", ".join(k for k, ok in checks.items() if not ok))

        trace = matrix.astype(np.int16)
        key_epoch = epochs.key_epoch.to_numpy(dtype=np.int64)
        np.savez_compressed(args.output_dir / f"{name}.npz", trace=trace, key_epoch=key_epoch)
        per_run.append({
            "run": name,
            "rekey_every": np.nan if name.startswith("idle_") else (0 if name.startswith("static_") else int(name[1:].split("_", 1)[0])),
            "run_type": "idle" if name.startswith("idle_") else ("static" if name.startswith("static_") else "rekeyed"),
            "rounds": matrix.shape[0],
            "active_rounds": summary["active_rounds"],
            "rotations": summary["rotations"],
            "probe_contention": int(matrix.sum()),
            "nonzero_sets": int(np.count_nonzero(matrix.sum(axis=0))),
            "victim_fills": summary["victim_fill_total"],
            "cross_pid_evictions": summary["L2_crossPID"],
            "probe_to_cross_pid_ratio": float(matrix.sum() / summary["L2_crossPID"]) if summary["L2_crossPID"] else np.nan,
            "walks": summary["victim_walks"],
            "sim_time_s": summary["sim_time_s"],
            "all_checks_pass": all(checks.values()),
            "deterministic_repeat": None,
        })

    groups = {}
    for name in matrices:
        if name.startswith("idle_"):
            condition = "idle"
        elif name.startswith("static_"):
            condition = "static"
        else:
            condition = name.split("_", 1)[0]
        groups.setdefault(condition, []).append(name)
    for names in groups.values():
        if len(names) < 2:
            continue
        first = names[0]
        deterministic = all(
            np.array_equal(matrices[first], matrices[name])
            and epoch_tables[first].equals(epoch_tables[name])
            for name in names[1:]
        )
        for item in per_run:
            if item["run"] in names:
                item["deterministic_repeat"] = deterministic
                item["all_checks_pass"] = item["all_checks_pass"] and deterministic
        if not deterministic:
            failures.append(f"{','.join(names)}: repeated CSVs differ")

    # --- Table 7.4 comparison: attacker "recoverable" contention ---------------
    static_names = [n for n in matrices if n.startswith("static_")]
    recover_rows = []
    if static_names:
        s_stack = np.mean([matrices[n].sum(axis=0) for n in static_names], axis=0)
        F0 = s_stack > 0
        ref_list = [matrices[n][:, F0].sum() / max(1, int(summaries[n]["active_rounds"]))
                    for n in static_names]
        ref = float(np.mean(ref_list)) if ref_list else 0.0
        for item in per_run:
            name = item["run"]
            if name.startswith("idle_"):
                continue
            m = matrices[name]
            ar = max(1, int(item["active_rounds"]))
            recov = m[:, F0].sum() / ar
            R = int(item["rotations"])
            recover_rows.append({
                "run": name, "rekey_every": item["rekey_every"], "rotations": R,
                "recoverable_per_round": recov,
                "recoverable_share": (recov / ref) if ref else np.nan,
                "model_1_over_Rp1": 1.0 / (R + 1),
                "model_with_floor": 1.0 / (R + 1) + (R / (R + 1)) * (int(F0.sum()) / 256.0),
            })
    recover = pd.DataFrame(recover_rows)
    rec_grp = pd.DataFrame()
    if not recover.empty:
        recover = recover.sort_values(["rekey_every", "run"])
        recover.to_csv(args.output_dir / "liverekeyprobe_recoverable.csv", index=False)
        rec_grp = recover[recover.rekey_every > 0].groupby("rekey_every", as_index=False).agg(
            rotations=("rotations", "first"),
            recoverable_share=("recoverable_share", "mean"),
            model_1_over_Rp1=("model_1_over_Rp1", "first"),
            model_with_floor=("model_with_floor", "first"))

    if not per_run:
        raise SystemExit("No complete run summaries found")
    runs = pd.DataFrame(per_run).sort_values(["rekey_every", "run"])
    runs.to_csv(args.output_dir / "liverekeyprobe_runs.csv", index=False)

    grouped = runs.groupby(["run_type", "rekey_every"], as_index=False, dropna=False).agg(
        repetitions=("run", "count"),
        mean_probe_contention=("probe_contention", "mean"),
        std_probe_contention=("probe_contention", "std"),
        mean_nonzero_sets=("nonzero_sets", "mean"),
        mean_cross_pid_evictions=("cross_pid_evictions", "mean"),
        mean_probe_to_cross_pid_ratio=("probe_to_cross_pid_ratio", "mean"),
        all_checks_pass=("all_checks_pass", "all"),
    )
    grouped.to_csv(args.output_dir / "liverekeyprobe_results.csv", index=False)
    lines = [
        "# Attacker-view live re-key results",
        "",
        "The primary matrix records attacker probe contention per round and set (0–8).",
        "Victim fills and L2 cross-PID eviction counters are retained separately for accounting.",
        "Each `.npz` contains `trace` as int16 `[R,256]` and `key_epoch` as `[R]`.",
        "All numeric results below come from completed simulator runs.",
        "",
        grouped.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "## Run checks",
        "",
        runs[["run", "deterministic_repeat", "all_checks_pass"]].to_string(index=False),
    ]
    if not rec_grp.empty:
        lines += ["", "## Attacker-recoverable contention vs Table 7.4", "",
                  "recoverable_share = mean per-round attacker probe contention inside the",
                  "static footprint F0, as a fraction of the static run -- the attacker",
                  "counterpart of Table 7.4's Recoverable column (model raw/(R+1), plus a",
                  "chance-overlap floor |F0|/256).", "",
                  rec_grp.to_string(index=False, float_format=lambda x: f"{x:.4f}")]
    if failures:
        lines += ["", "## Failed checks", "", *[f"- {item}" for item in failures]]
    (args.output_dir / "liverekeyprobe_results.md").write_text("\n".join(lines) + "\n")

    plot = grouped[grouped.rekey_every > 0].sort_values("rekey_every")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.errorbar(plot.rekey_every, plot.mean_probe_contention, yerr=plot.std_probe_contention.fillna(0), marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Re-key period k (rounds)")
    ax.set_ylabel("Total attacker probe contention")
    ax.set_title("Attacker-view contention under live re-keying")
    fig.tight_layout()
    fig.savefig(args.output_dir / "liverekeyprobe_contention.png", dpi=160)

    if not rec_grp.empty:
        fig2, ax2 = plt.subplots(figsize=(7.8, 4.8))
        ax2.plot(rec_grp.rekey_every, rec_grp.recoverable_share, "o-", label="Measured (attacker probe)")
        ax2.plot(rec_grp.rekey_every, rec_grp.model_1_over_Rp1, "s--", label="Table 7.4  1/(R+1)")
        ax2.plot(rec_grp.rekey_every, rec_grp.model_with_floor, "^:", label="With chance floor")
        ax2.set_xscale("log", base=2)
        ax2.set_xlabel("Re-key period k (rounds)")
        ax2.set_ylabel("Recoverable share vs static")
        ax2.set_title("Attacker-recoverable contention vs Table 7.4")
        ax2.legend(); fig2.tight_layout()
        fig2.savefig(args.output_dir / "liverekeyprobe_vs_table74.png", dpi=160)
    print(grouped.to_string(index=False))
    print(f"checks: {sum(r['all_checks_pass'] for r in per_run)}/{len(per_run)} runs pass; failures: {failures or 'none'}")


if __name__ == "__main__":
    main()
