# -*- coding: utf-8 -*-
"""问题3候选改进的2月与连续60天公平对照。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from problem3_cost_diagnostics import attribute_unused
from solve_problem3 import (
    LOAD_FORECAST_MODES,
    RunConfig,
    StorageParams,
    forecast_load,
    make_targets,
    read_problem_data,
    simulate_strategy,
    validate_run,
)


START_DAY = date(2025, 2, 1)
END_60_DAY = date(2025, 4, 1)
FEB_END = date(2025, 2, 28)


def forecast_accuracy(data, history_days: int) -> pd.DataFrame:
    rows = []
    scopes = {
        "feb2025": (pd.Timestamp(START_DAY), pd.Timestamp(FEB_END)),
        "60days": (pd.Timestamp(START_DAY), pd.Timestamp(END_60_DAY)),
    }
    records: list[dict] = []
    for day in pd.date_range(START_DAY, END_60_DAY, freq="D"):
        for hour in (0, 6, 12, 18):
            decision_time = day + pd.Timedelta(hours=hour)
            targets = make_targets(decision_time)
            actual = data.actual_vector(targets, "load")
            for mode in LOAD_FORECAST_MODES:
                predicted = forecast_load(
                    data, decision_time, targets, history_days, mode
                )
                for horizon_name, count in (
                    ("next_committed_6h", min(36, len(targets))),
                    ("available_24h", len(targets)),
                ):
                    error = predicted[:count] - actual[:count]
                    finite = np.isfinite(error)
                    for value in error[finite]:
                        records.append(
                            {
                                "date": day,
                                "update_hour": hour,
                                "forecast_mode": mode,
                                "horizon": horizon_name,
                                "error_kwh": float(value),
                            }
                        )
    frame = pd.DataFrame(records)
    for scope, (start, end) in scopes.items():
        selected = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        for keys, group in selected.groupby(
            ["forecast_mode", "update_hour", "horizon"], sort=False
        ):
            mode, hour, horizon = keys
            error = group["error_kwh"].to_numpy(float)
            rows.append(
                {
                    "scope": scope,
                    "forecast_mode": mode,
                    "update_hour": int(hour),
                    "horizon": horizon,
                    "observations": int(len(error)),
                    "mae_kwh_per_10min": float(np.mean(np.abs(error))),
                    "rmse_kwh_per_10min": float(np.sqrt(np.mean(error**2))),
                    "bias_kwh_per_10min": float(np.mean(error)),
                }
            )
        for (mode, horizon), group in selected.groupby(
            ["forecast_mode", "horizon"], sort=False
        ):
            error = group["error_kwh"].to_numpy(float)
            rows.append(
                {
                    "scope": scope,
                    "forecast_mode": mode,
                    "update_hour": "all",
                    "horizon": horizon,
                    "observations": int(len(error)),
                    "mae_kwh_per_10min": float(np.mean(np.abs(error))),
                    "rmse_kwh_per_10min": float(np.sqrt(np.mean(error**2))),
                    "bias_kwh_per_10min": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def summarize_scope(
    run: dict[str, object],
    checks: dict[str, object],
    params: StorageParams,
    label: str,
    module: str,
    scope: str,
    start: date,
    end: date,
    initial_soc: float,
) -> dict[str, object]:
    detail = run["detail"]
    daily = run["daily"]
    assert isinstance(detail, pd.DataFrame) and isinstance(daily, pd.DataFrame)
    dates = pd.to_datetime(detail["date"])
    selected = detail[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()
    daily_dates = pd.to_datetime(daily["date"])
    selected_daily = daily[
        (daily_dates >= pd.Timestamp(start)) & (daily_dates <= pd.Timestamp(end))
    ].copy()
    final_contract_cost = float(
        (selected["price_yuan_per_kwh"] * selected["final_purchase_kwh"]).sum()
    )
    adjustment_deviation = float(
        (
            0.5
            * selected["price_yuan_per_kwh"]
            * (
                selected["upward_adjustment_kwh"]
                + selected["downward_adjustment_kwh"]
            )
        ).sum()
    )
    _, unused_summary = attribute_unused(selected, params)
    daily_peak = selected.groupby("date")["soc_end_kwh"].max()
    total = float(selected["total_cost_yuan"].sum())
    return {
        "scope": scope,
        "candidate": label,
        "changed_module": module,
        "load_forecast_mode": str(selected["load_forecast_mode"].iloc[0]),
        "day_ahead_mode": str(selected["day_ahead_mode"].iloc[0]),
        "terminal_value_mode": str(selected["terminal_value_mode"].iloc[0]),
        "start_date": str(start),
        "end_date": str(end),
        "days": int(selected_daily["date"].nunique()),
        "initial_soc_kwh": initial_soc,
        "total_cost_yuan": total,
        "final_contract_normal_cost_yuan": final_contract_cost,
        "adjustment_deviation_cost_yuan": adjustment_deviation,
        "emergency_cost_yuan": float(selected["emergency_cost_yuan"].sum()),
        "throughput_penalty_yuan": float(selected["throughput_penalty_yuan"].sum()),
        "upward_adjustment_kwh": float(selected["upward_adjustment_kwh"].sum()),
        "downward_adjustment_kwh": float(selected["downward_adjustment_kwh"].sum()),
        "total_adjustment_kwh": float(
            selected["upward_adjustment_kwh"].sum()
            + selected["downward_adjustment_kwh"].sum()
        ),
        "emergency_purchase_kwh": float(selected["emergency_purchase_kwh"].sum()),
        "unused_kwh": float(selected["curtailment_or_unused_kwh"].sum()),
        "avoidable_contract_unused_kwh": float(
            unused_summary["source_attribution_non_overlapping"]
            ["avoidable_contract_unused_kwh"]
        ),
        "avoidable_contract_unused_cost_yuan": float(
            unused_summary["source_attribution_non_overlapping"]
            ["avoidable_contract_cost_yuan"]
        ),
        "storage_throughput_kwh": float(
            selected["charge_kwh"].sum() + selected["discharge_kwh"].sum()
        ),
        "mean_daily_ending_soc_kwh": float(selected_daily["soc_end_kwh"].mean()),
        "days_reaching_soc_max": int((daily_peak >= params.soc_max - 1.0e-7).sum()),
        "ending_soc_kwh": float(selected_daily["soc_end_kwh"].iloc[-1]),
        "runtime_seconds_full_60day_run": float(run["runtime_seconds"]),
        "all_checks_passed": bool(checks["all_checks_passed"]),
        "emergency_while_charging_count": int(
            (
                (selected["emergency_purchase_kwh"] > 1.0e-7)
                & (selected["charge_kwh"] > 1.0e-7)
            ).sum()
        ),
        "unused_while_discharging_count": int(
            (
                (selected["curtailment_or_unused_kwh"] > 1.0e-7)
                & (selected["discharge_kwh"] > 1.0e-7)
            ).sum()
        ),
        "max_balance_residual_kwh": float(selected["balance_residual_kwh"].abs().max()),
        "max_soc_residual_kwh": float(selected["soc_residual_kwh"].abs().max()),
        "historical_lock_ok": bool(checks["historical_decision_lock_ok"]),
        "mature_history_only": bool(checks["mature_history_only"]),
    }


def run_candidate(data, params, base_config, spec, initial_soc):
    label, module, load_mode, day_ahead_mode, terminal_mode = spec
    config = replace(
        base_config,
        load_forecast_mode=load_mode,
        day_ahead_mode=day_ahead_mode,
        terminal_value_mode=terminal_mode,
    )
    print(f"[候选测试] {label}", flush=True)
    run = simulate_strategy(
        data=data,
        start_day=START_DAY,
        end_day=END_60_DAY,
        strategy="S061218",
        initial_soc=initial_soc,
        params=params,
        config=config,
        progress_every_days=15,
        control_policy="updated_forecast_rolling",
    )
    checks = validate_run(run, params, config)
    return run, checks


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
        default=Path("analysis_outputs/problem3_cost_improvement_tests"),
    )
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--soc-step", type=float, default=100.0)
    args = parser.parse_args()
    baseline_detail = pd.read_csv(args.baseline_run_dir / "problem3_detail.csv")
    baseline_detail["date"] = pd.to_datetime(baseline_detail["date"])
    initial_row = baseline_detail[baseline_detail["date"] == pd.Timestamp(START_DAY)].iloc[0]
    initial_soc = float(initial_row["soc_start_kwh"])
    params = StorageParams()
    data = read_problem_data(args.data_dir)
    base_config = RunConfig(
        scenario_count=args.scenarios,
        history_days=args.history_days,
        seed=args.seed,
        soc_step=args.soc_step,
        feedback_policy="grid-boundary",
        forbid_emergency_charging=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy = forecast_accuracy(data, args.history_days)
    accuracy.to_csv(
        args.output_dir / "load_forecast_accuracy.csv", index=False, encoding="utf-8-sig"
    )

    forecast_specs = [
        (
            "current_model" if mode == "current" else f"forecast_{mode}",
            "current" if mode == "current" else "load_forecast_only",
            mode,
            "current",
            "fixed-linear",
        )
        for mode in LOAD_FORECAST_MODES
    ]
    results: list[dict[str, object]] = []
    runs: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for spec in forecast_specs:
        run, checks = run_candidate(data, params, base_config, spec, initial_soc)
        runs[spec[0]] = (run, checks)
        results.append(
            summarize_scope(run, checks, params, spec[0], spec[1], "feb2025", START_DAY, FEB_END, initial_soc)
        )
        results.append(
            summarize_scope(run, checks, params, spec[0], spec[1], "60days", START_DAY, END_60_DAY, initial_soc)
        )
    forecast_60 = pd.DataFrame(results)
    forecast_60 = forecast_60[
        (forecast_60["scope"] == "60days")
        & (forecast_60["changed_module"].isin(["current", "load_forecast_only"]))
    ]
    best_forecast_label = str(
        forecast_60.sort_values("total_cost_yuan").iloc[0]["candidate"]
    )
    best_forecast_mode = str(
        forecast_60.sort_values("total_cost_yuan").iloc[0]["load_forecast_mode"]
    )
    extra_specs = [
        ("recourse_aware_only", "day_ahead_only", "current", "recourse-aware", "fixed-linear"),
        ("cross_midnight_dp_only", "terminal_value_only", "current", "current", "cross-midnight-dp"),
        ("state_dependent_terminal_only", "terminal_value_only", "current", "current", "state-dependent"),
        ("best_forecast_plus_recourse", "load_plus_day_ahead", best_forecast_mode, "recourse-aware", "fixed-linear"),
        ("best_forecast_plus_recourse_plus_state", "all_three", best_forecast_mode, "recourse-aware", "state-dependent"),
    ]
    for spec in extra_specs:
        run, checks = run_candidate(data, params, base_config, spec, initial_soc)
        runs[spec[0]] = (run, checks)
        results.append(
            summarize_scope(run, checks, params, spec[0], spec[1], "feb2025", START_DAY, FEB_END, initial_soc)
        )
        results.append(
            summarize_scope(run, checks, params, spec[0], spec[1], "60days", START_DAY, END_60_DAY, initial_soc)
        )

    comparison = pd.DataFrame(results)
    baseline = comparison[comparison["candidate"] == "current_model"].set_index("scope")
    for scope, frame in comparison.groupby("scope"):
        base = baseline.loc[scope]
        mask = comparison["scope"] == scope
        comparison.loc[mask, "cost_saving_vs_current_yuan"] = (
            float(base["total_cost_yuan"])
            - comparison.loc[mask, "total_cost_yuan"]
        )
        comparison.loc[mask, "final_contract_cost_change_yuan"] = (
            comparison.loc[mask, "final_contract_normal_cost_yuan"]
            - float(base["final_contract_normal_cost_yuan"])
        )
        comparison.loc[mask, "adjustment_cost_change_yuan"] = (
            comparison.loc[mask, "adjustment_deviation_cost_yuan"]
            - float(base["adjustment_deviation_cost_yuan"])
        )
        comparison.loc[mask, "emergency_cost_change_yuan"] = (
            comparison.loc[mask, "emergency_cost_yuan"]
            - float(base["emergency_cost_yuan"])
        )
        comparison.loc[mask, "throughput_cost_change_yuan"] = (
            comparison.loc[mask, "throughput_penalty_yuan"]
            - float(base["throughput_penalty_yuan"])
        )
    comparison.to_csv(
        args.output_dir / "cost_improvement_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    checks_payload = {
        label: checks for label, (_, checks) in runs.items()
    }
    checks_payload["selected_best_forecast_from_60day_diagnostic"] = {
        "candidate": best_forecast_label,
        "mode": best_forecast_mode,
        "selection_warning": "Diagnostic-period selection; not a future formal-policy claim.",
    }
    (args.output_dir / "candidate_checks.json").write_text(
        json.dumps(checks_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
