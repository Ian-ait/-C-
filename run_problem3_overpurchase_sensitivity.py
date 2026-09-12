# -*- coding: utf-8 -*-
"""问题3 g0分区减购敏感性：5个种子、2月与连续60天。"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from solve_problem3 import RunConfig, StorageParams, read_problem_data, simulate_strategy, validate_run


START = date(2025, 2, 1)
FEB_END = date(2025, 2, 28)
END_60 = date(2025, 4, 1)
REGIONS = ("00-06", "06-24", "all-day")
FACTORS = (1.00, 0.98, 0.96, 0.94, 0.92, 0.90)
SEEDS = (2025, 2026, 2027, 2028, 2029)

_DATA = None
_PARAMS = None
_INITIAL_SOC = None
_OUTPUT_DIR = None


def initialize_worker(data_dir: str, initial_soc: float, output_dir: str) -> None:
    global _DATA, _PARAMS, _INITIAL_SOC, _OUTPUT_DIR
    _DATA = read_problem_data(Path(data_dir))
    _PARAMS = StorageParams()
    _INITIAL_SOC = float(initial_soc)
    _OUTPUT_DIR = Path(output_dir)


def summarize(detail: pd.DataFrame, daily: pd.DataFrame, start: date, end: date) -> dict:
    dates = pd.to_datetime(detail["date"])
    frame = detail[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    daily_dates = pd.to_datetime(daily["date"])
    days = daily[(daily_dates >= pd.Timestamp(start)) & (daily_dates <= pd.Timestamp(end))]
    p = frame["price_yuan_per_kwh"]
    up_loss = 0.5 * p * frame["upward_adjustment_kwh"]
    down_loss = 0.5 * p * frame["downward_adjustment_kwh"]
    first_block = frame["slot"] < 36
    daily_peak = frame.groupby("date")["soc_end_kwh"].max()
    return {
        "start_date": str(start),
        "end_date": str(end),
        "days": int(days["date"].nunique()),
        "total_cost_yuan": float(frame["total_cost_yuan"].sum()),
        "normal_contract_cost_yuan": float((p * frame["final_purchase_kwh"]).sum()),
        "adjustment_deviation_cost_yuan": float((up_loss + down_loss).sum()),
        "upward_deviation_loss_yuan": float(up_loss.sum()),
        "downward_deviation_loss_yuan": float(down_loss.sum()),
        "emergency_cost_yuan": float(frame["emergency_cost_yuan"].sum()),
        "throughput_penalty_yuan": float(frame["throughput_penalty_yuan"].sum()),
        "base_purchase_cost_yuan": float(frame["base_purchase_cost_yuan"].sum()),
        "downward_refund_yuan": float(frame["downward_refund_yuan"].sum()),
        "upward_premium_yuan": float(frame["upward_premium_yuan"].sum()),
        "g0_purchase_kwh": float(frame["g0_kwh"].sum()),
        "final_contract_purchase_kwh": float(frame["final_purchase_kwh"].sum()),
        "upward_adjustment_kwh": float(frame["upward_adjustment_kwh"].sum()),
        "downward_adjustment_kwh": float(frame["downward_adjustment_kwh"].sum()),
        "total_adjustment_kwh": float(
            frame["upward_adjustment_kwh"].sum() + frame["downward_adjustment_kwh"].sum()
        ),
        "emergency_purchase_kwh": float(frame["emergency_purchase_kwh"].sum()),
        "unused_kwh": float(frame["curtailment_or_unused_kwh"].sum()),
        "first_block_unused_kwh": float(
            frame.loc[first_block, "curtailment_or_unused_kwh"].sum()
        ),
        "first_block_emergency_kwh": float(
            frame.loc[first_block, "emergency_purchase_kwh"].sum()
        ),
        "mean_daily_ending_soc_kwh": float(days["soc_end_kwh"].mean()),
        "days_reaching_soc_max": int((daily_peak >= 10800.0 - 1.0e-7).sum()),
        "ending_soc_kwh": float(days["soc_end_kwh"].iloc[-1]),
        "max_balance_residual_kwh": float(frame["balance_residual_kwh"].abs().max()),
        "max_soc_residual_kwh": float(frame["soc_residual_kwh"].abs().max()),
        "emergency_while_charging_count": int(
            ((frame["emergency_purchase_kwh"] > 1.0e-7) & (frame["charge_kwh"] > 1.0e-7)).sum()
        ),
        "unused_while_discharging_count": int(
            ((frame["curtailment_or_unused_kwh"] > 1.0e-7) & (frame["discharge_kwh"] > 1.0e-7)).sum()
        ),
    }


def output_path(seed: int, region: str, factor: float) -> Path:
    assert _OUTPUT_DIR is not None
    return _OUTPUT_DIR / f"seed_{seed}" / f"region_{region}" / f"factor_{factor:.2f}"


def run_one(task: tuple[int, str, float]) -> list[dict]:
    seed, region, factor = task
    assert _DATA is not None and _PARAMS is not None and _INITIAL_SOC is not None
    config = RunConfig(
        scenario_count=12,
        history_days=30,
        seed=seed,
        soc_step=100.0,
        feedback_policy="grid-boundary",
        forbid_emergency_charging=True,
        load_forecast_mode="current",
        day_ahead_mode="current",
        terminal_value_mode="fixed-linear",
        g0_scale_region=region,
        g0_scale_factor=factor,
    )
    run = simulate_strategy(
        data=_DATA,
        start_day=START,
        end_day=END_60,
        strategy="S061218",
        initial_soc=_INITIAL_SOC,
        params=_PARAMS,
        config=config,
        progress_every_days=60,
        control_policy="updated_forecast_rolling",
    )
    checks = validate_run(run, _PARAMS, config)
    detail = run["detail"]
    daily = run["daily"]
    assert isinstance(detail, pd.DataFrame) and isinstance(daily, pd.DataFrame)
    directory = output_path(seed, region, factor)
    directory.mkdir(parents=True, exist_ok=True)
    daily.to_csv(directory / "daily_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(directory / "problem3_detail.csv", index=False, encoding="utf-8-sig")
    (directory / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for scope, start, end in (
        ("feb2025", START, FEB_END),
        ("60days", START, END_60),
    ):
        row = summarize(detail, daily, start, end)
        row.update(
            {
                "scope": scope,
                "seed": seed,
                "scale_region": region,
                "scale_factor": factor,
                "initial_soc_kwh": _INITIAL_SOC,
                "runtime_seconds": float(run["runtime_seconds"]),
                "all_checks_passed": bool(checks["all_checks_passed"]),
                "mature_history_only": bool(checks["mature_history_only"]),
                "historical_lock_ok": bool(checks["historical_decision_lock_ok"]),
            }
        )
        rows.append(row)
    (directory / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def add_economics(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[
        (results["scale_factor"] == 1.0)
        & (results["scale_region"] == "all-day")
    ].set_index(["scope", "seed"])
    metric_pairs = {
        "normal_purchase_saving_yuan": "normal_contract_cost_yuan",
        "downward_loss_saving_yuan": "downward_deviation_loss_yuan",
        "upward_loss_saving_yuan": "upward_deviation_loss_yuan",
        "unused_reduction_kwh": "unused_kwh",
        "first_block_unused_reduction_kwh": "first_block_unused_kwh",
        "g0_purchase_reduction_kwh": "g0_purchase_kwh",
    }
    rows = []
    for row in results.to_dict("records"):
        base = baseline.loc[(row["scope"], row["seed"])]
        for output, metric in metric_pairs.items():
            row[output] = float(base[metric]) - float(row[metric])
        row["incremental_upward_loss_yuan"] = (
            float(row["upward_deviation_loss_yuan"])
            - float(base["upward_deviation_loss_yuan"])
        )
        row["incremental_emergency_cost_yuan"] = (
            float(row["emergency_cost_yuan"]) - float(base["emergency_cost_yuan"])
        )
        row["incremental_emergency_kwh"] = (
            float(row["emergency_purchase_kwh"])
            - float(base["emergency_purchase_kwh"])
        )
        row["incremental_throughput_cost_yuan"] = (
            float(row["throughput_penalty_yuan"])
            - float(base["throughput_penalty_yuan"])
        )
        row["net_benefit_yuan"] = float(base["total_cost_yuan"]) - float(row["total_cost_yuan"])
        row["net_benefit_reconstruction_yuan"] = (
            row["normal_purchase_saving_yuan"]
            + row["downward_loss_saving_yuan"]
            - row["incremental_upward_loss_yuan"]
            - row["incremental_emergency_cost_yuan"]
            - row["incremental_throughput_cost_yuan"]
        )
        reduction = row["unused_reduction_kwh"]
        row["normal_saving_per_unused_kwh_yuan"] = (
            row["normal_purchase_saving_yuan"] / reduction if reduction > 1.0e-9 else np.nan
        )
        added_emergency = row["incremental_emergency_kwh"]
        row["incremental_cost_per_added_emergency_kwh_yuan"] = (
            row["incremental_emergency_cost_yuan"] / added_emergency
            if added_emergency > 1.0e-9 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "total_cost_yuan", "normal_contract_cost_yuan", "adjustment_deviation_cost_yuan",
        "emergency_cost_yuan", "unused_kwh", "upward_adjustment_kwh",
        "downward_adjustment_kwh", "first_block_unused_kwh", "first_block_emergency_kwh",
        "mean_daily_ending_soc_kwh", "days_reaching_soc_max", "net_benefit_yuan",
        "normal_purchase_saving_yuan", "downward_loss_saving_yuan",
        "incremental_upward_loss_yuan", "incremental_emergency_cost_yuan",
        "unused_reduction_kwh", "incremental_emergency_kwh",
    ]
    rows = []
    for keys, group in results.groupby(["scope", "scale_region", "scale_factor"]):
        scope, region, factor = keys
        row = {
            "scope": scope,
            "scale_region": region,
            "scale_factor": factor,
            "seed_count": int(group["seed"].nunique()),
            "all_checks_passed": bool(group["all_checks_passed"].all()),
            "all_mature_history_only": bool(group["mature_history_only"].all()),
            "all_historical_locks_ok": bool(group["historical_lock_ok"].all()),
            "max_balance_residual_kwh": float(group["max_balance_residual_kwh"].max()),
            "max_soc_residual_kwh": float(group["max_soc_residual_kwh"].max()),
            "total_emergency_while_charging_count": int(group["emergency_while_charging_count"].sum()),
            "total_unused_while_discharging_count": int(group["unused_while_discharging_count"].sum()),
            "seeds_with_cost_saving": int((group["net_benefit_yuan"] > 0).sum()),
            "worst_seed_net_benefit_yuan": float(group["net_benefit_yuan"].min()),
            "cost_standard_deviation_yuan": float(group["total_cost_yuan"].std(ddof=0)),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(group[metric].mean())
        row["economic_test_passed"] = bool(
            row["mean_net_benefit_yuan"] > 0
            and row["worst_seed_net_benefit_yuan"] >= 0
            and row["mean_incremental_emergency_cost_yuan"]
            < row["mean_normal_purchase_saving_yuan"] + row["mean_downward_loss_saving_yuan"]
            and row["all_checks_passed"]
            and row["all_mature_history_only"]
            and row["all_historical_locks_ok"]
            and row["total_emergency_while_charging_count"] == 0
            and row["total_unused_while_discharging_count"] == 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


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
        default=Path("analysis_outputs/problem3_overpurchase_sensitivity"),
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    baseline = pd.read_csv(args.baseline_run_dir / "problem3_detail.csv")
    baseline["date"] = pd.to_datetime(baseline["date"])
    initial_soc = float(
        baseline[baseline["date"] == pd.Timestamp(START)]["soc_start_kwh"].iloc[0]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in SEEDS:
        tasks.append((seed, "all-day", 1.0))
        for region in REGIONS:
            for factor in FACTORS[1:]:
                tasks.append((seed, region, factor))
    result_rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.data_dir), initial_soc, str(args.output_dir)),
    ) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            result_rows.extend(future.result())
            print(f"[敏感性进度] {completed}/{len(tasks)} 完成 {task}", flush=True)
    raw = pd.DataFrame(result_rows)
    baseline_rows = raw[raw["scale_factor"] == 1.0].copy()
    replicated = []
    for region in REGIONS:
        copy = baseline_rows.copy()
        copy["scale_region"] = region
        replicated.append(copy)
        for seed in SEEDS:
            directory = args.output_dir / f"seed_{seed}" / f"region_{region}" / "factor_1.00"
            directory.mkdir(parents=True, exist_ok=True)
            source = args.output_dir / f"seed_{seed}" / "region_all-day" / "factor_1.00"
            if directory != source:
                for filename in ("daily_summary.csv", "problem3_detail.csv", "checks.json"):
                    shutil.copy2(source / filename, directory / filename)
            rows = copy[copy["seed"] == seed].to_dict("records")
            (directory / "summary.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    raw = pd.concat([raw[raw["scale_factor"] != 1.0], *replicated], ignore_index=True)
    detailed = add_economics(raw)
    summary = aggregate(detailed)
    detailed.to_csv(
        args.output_dir / "g0_scaling_results.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "g0_scaling_summary.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "initial_soc_kwh": initial_soc,
        "seeds": list(SEEDS),
        "regions": list(REGIONS),
        "factors": list(FACTORS),
        "baseline_factor_1_reused_across_regions": True,
        "run_dates": [str(START), str(END_60)],
        "formal_result3_written": False,
        "default_strategy_changed": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.sort_values(["scope", "mean_total_cost_yuan"]).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
