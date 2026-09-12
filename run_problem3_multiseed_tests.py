# -*- coding: utf-8 -*-
"""问题3候选改进的多随机种子验证。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from run_problem3_improvement_tests import (
    FEB_END,
    START_DAY,
    END_60_DAY,
    run_candidate,
    summarize_scope,
)
from solve_problem3 import RunConfig, StorageParams, read_problem_data


CANDIDATE_SPECS = [
    ("current_model", "current", "current", "current", "fixed-linear"),
    ("observed_corrected_only", "load_forecast_only", "observed-corrected", "current", "fixed-linear"),
    ("recourse_aware_only", "day_ahead_only", "current", "recourse-aware", "fixed-linear"),
    ("cross_midnight_dp_only", "terminal_value_only", "current", "current", "cross-midnight-dp"),
    ("state_dependent_terminal_only", "terminal_value_only", "current", "current", "state-dependent"),
    ("observed_corrected_plus_recourse", "load_plus_day_ahead", "observed-corrected", "recourse-aware", "fixed-linear"),
    ("observed_corrected_plus_recourse_plus_state", "all_three", "observed-corrected", "recourse-aware", "state-dependent"),
]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("至少需要一个随机种子")
    if len(set(seeds)) != len(seeds):
        raise ValueError("随机种子不能重复")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("附件"))
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_control_compare_full_year/updated_forecast_rolling"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_multiseed_improvement_tests"),
    )
    parser.add_argument("--seeds", default="2025,2026,2027,2028,2029")
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--soc-step", type=float, default=100.0)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    baseline = pd.read_csv(args.baseline_run_dir / "problem3_detail.csv")
    baseline["date"] = pd.to_datetime(baseline["date"])
    initial_row = baseline[baseline["date"] == pd.Timestamp(START_DAY)].iloc[0]
    initial_soc = float(initial_row["soc_start_kwh"])
    data = read_problem_data(args.data_dir)
    params = StorageParams()
    rows: list[dict] = []

    for seed in seeds:
        base_config = RunConfig(
            scenario_count=args.scenarios,
            history_days=args.history_days,
            seed=seed,
            soc_step=args.soc_step,
            feedback_policy="grid-boundary",
            forbid_emergency_charging=True,
        )
        for label, module, load_mode, day_ahead_mode, terminal_mode in CANDIDATE_SPECS:
            spec = (label, module, load_mode, day_ahead_mode, terminal_mode)
            run, checks = run_candidate(data, params, base_config, spec, initial_soc)
            for scope, start, end in (
                ("feb2025", START_DAY, FEB_END),
                ("60days", START_DAY, END_60_DAY),
            ):
                result = summarize_scope(
                    run, checks, params, label, module, scope, start, end, initial_soc
                )
                result["seed"] = seed
                rows.append(result)

    detailed = pd.DataFrame(rows)
    detailed["cost_saving_vs_seed_baseline_yuan"] = 0.0
    for (seed, scope), group in detailed.groupby(["seed", "scope"]):
        baseline_cost = float(
            group.loc[group["candidate"] == "current_model", "total_cost_yuan"].iloc[0]
        )
        mask = (detailed["seed"] == seed) & (detailed["scope"] == scope)
        detailed.loc[mask, "cost_saving_vs_seed_baseline_yuan"] = (
            baseline_cost - detailed.loc[mask, "total_cost_yuan"]
        )

    summary_metrics = [
        "total_cost_yuan",
        "final_contract_normal_cost_yuan",
        "adjustment_deviation_cost_yuan",
        "emergency_cost_yuan",
        "total_adjustment_kwh",
        "upward_adjustment_kwh",
        "downward_adjustment_kwh",
        "emergency_purchase_kwh",
        "unused_kwh",
        "avoidable_contract_unused_cost_yuan",
        "storage_throughput_kwh",
        "cost_saving_vs_seed_baseline_yuan",
    ]
    summary = (
        detailed.groupby(["scope", "candidate"], as_index=False)[summary_metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part) != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "candidate_results_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "seed_summary.csv", index=False, encoding="utf-8-sig")
    detailed.to_csv(args.output_dir / "cost_improvement_comparison.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
