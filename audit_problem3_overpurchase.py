# -*- coding: utf-8 -*-
"""问题3正式结果的合同过购与无法利用电量只读审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solve_problem3 import RunConfig, StorageParams, build_scenarios, read_problem_data


FORMAL_START = pd.Timestamp("2025-02-01")
FORMAL_END = pd.Timestamp("2025-12-31")
TOL = 1.0e-7


def time_block(slot: pd.Series) -> pd.Series:
    return pd.cut(
        slot,
        bins=[-1, 35, 71, 107, 143],
        labels=["00:00-06:00", "06:00-12:00", "12:00-18:00", "18:00-24:00"],
    ).astype(str)


def build_detail(detail: pd.DataFrame, params: StorageParams) -> tuple[pd.DataFrame, dict]:
    frame = detail.reset_index(drop=True).copy()
    w = frame["curtailment_or_unused_kwh"].to_numpy(float)
    g = frame["final_purchase_kwh"].to_numpy(float)
    g0 = frame["g0_kwh"].to_numpy(float)
    up = frame["upward_adjustment_kwh"].to_numpy(float)
    pv = frame["actual_pv_kwh"].to_numpy(float)
    load = frame["actual_load_kwh"].to_numpy(float)
    retained_g0 = np.minimum(g0, g)
    total_source = retained_g0 + up + pv
    source_balance = np.max(np.abs(total_source - (g + pv)))
    if source_balance > 1.0e-6:
        raise RuntimeError(f"合同来源分解不闭合：{source_balance}")

    soc_cap = np.maximum(
        (params.soc_max - frame["soc_start_kwh"].to_numpy(float))
        / params.eta_charge,
        0.0,
    )
    power_cap = np.full(len(frame), params.max_charge_kwh)
    absorb_cap = np.minimum(soc_cap, power_cap)
    surplus = np.maximum(g + pv - load, 0.0)
    physical_forced = np.maximum(surplus - absorb_cap, 0.0)
    policy_gap = np.maximum(w - physical_forced, 0.0)

    contract_lb = np.maximum(w - pv, 0.0)
    contract_ub = np.minimum(w, g)
    pv_allocation_lb = np.maximum(w - g, 0.0)
    pv_allocation_ub = np.minimum(w, pv)
    pv_unavoidable = np.maximum(pv - load - absorb_cap, 0.0)
    pv_unavoidable = np.minimum(pv_unavoidable, w)
    if np.max(contract_lb + pv_unavoidable - w) > 1.0e-6:
        raise RuntimeError("合同确定下界与光伏不可避免量不能共同成立")
    ambiguous = np.maximum(w - contract_lb - pv_unavoidable, 0.0)

    soc_effect = np.maximum(
        physical_forced - np.maximum(surplus - power_cap, 0.0), 0.0
    )
    power_effect = np.maximum(
        physical_forced - np.maximum(surplus - soc_cap, 0.0), 0.0
    )
    source_amounts = {
        "retained_g0": retained_g0,
        "upward_adjustment": up,
        "pv": pv,
    }
    for name, amount in source_amounts.items():
        frame[f"{name}_unused_lower_bound_kwh"] = np.maximum(
            w - (total_source - amount), 0.0
        )
        frame[f"{name}_unused_upper_bound_kwh"] = np.minimum(w, amount)

    frame["time_block"] = time_block(frame["slot"])
    frame["month"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m")
    quantiles = np.unique(
        np.quantile(frame["price_yuan_per_kwh"].to_numpy(float), [0, 0.25, 0.5, 0.75, 1])
    )
    if len(quantiles) < 3:
        quantiles = np.linspace(
            frame["price_yuan_per_kwh"].min(),
            frame["price_yuan_per_kwh"].max() + 1.0e-9,
            5,
        )
    quantiles[-1] += 1.0e-9
    frame["price_band"] = pd.cut(
        frame["price_yuan_per_kwh"], bins=quantiles, include_lowest=True
    ).astype(str)
    frame["soc_reaches_upper_bound"] = (
        (frame["soc_start_kwh"] >= params.soc_max - TOL)
        | (frame["soc_end_kwh"] >= params.soc_max - TOL)
    )
    frame["retained_g0_kwh"] = retained_g0
    frame["contract_unused_lower_bound_kwh"] = contract_lb
    frame["contract_unused_upper_bound_kwh"] = contract_ub
    frame["pv_unused_allocation_lower_bound_kwh"] = pv_allocation_lb
    frame["pv_unused_allocation_upper_bound_kwh"] = pv_allocation_ub
    frame["pv_unavoidable_unused_kwh"] = pv_unavoidable
    frame["joint_ambiguous_unused_kwh"] = ambiguous
    frame["soc_limit_effect_kwh_counterfactual"] = soc_effect
    frame["charge_power_limit_effect_kwh_counterfactual"] = power_effect
    frame["not_absorbed_despite_physical_capacity_kwh"] = policy_gap
    frame["contract_unused_cost_lower_bound_yuan"] = (
        contract_lb * frame["price_yuan_per_kwh"].to_numpy(float)
    )
    frame["contract_unused_cost_upper_bound_yuan"] = (
        contract_ub * frame["price_yuan_per_kwh"].to_numpy(float)
    )
    frame["source_attribution_reconstruction_residual_kwh"] = (
        w - contract_lb - pv_unavoidable - ambiguous
    )
    price_edges = [float(value) for value in quantiles]
    return frame, {"price_band_edges_yuan_per_kwh": price_edges}


SUM_COLUMNS = [
    "curtailment_or_unused_kwh",
    "contract_unused_lower_bound_kwh",
    "contract_unused_upper_bound_kwh",
    "pv_unused_allocation_lower_bound_kwh",
    "pv_unused_allocation_upper_bound_kwh",
    "pv_unavoidable_unused_kwh",
    "joint_ambiguous_unused_kwh",
    "soc_limit_effect_kwh_counterfactual",
    "charge_power_limit_effect_kwh_counterfactual",
    "not_absorbed_despite_physical_capacity_kwh",
    "contract_unused_cost_lower_bound_yuan",
    "contract_unused_cost_upper_bound_yuan",
    "retained_g0_unused_lower_bound_kwh",
    "retained_g0_unused_upper_bound_kwh",
    "upward_adjustment_unused_lower_bound_kwh",
    "upward_adjustment_unused_upper_bound_kwh",
    "pv_unused_lower_bound_kwh",
    "pv_unused_upper_bound_kwh",
]


def summarize_dimension(frame: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    grouped = frame.groupby(column, dropna=False)[SUM_COLUMNS].sum().reset_index()
    grouped.insert(0, "dimension", name)
    grouped = grouped.rename(columns={column: "dimension_value"})
    counts = (
        frame.assign(_positive=frame["curtailment_or_unused_kwh"] > TOL)
        .groupby(column, dropna=False)["_positive"]
        .agg(["size", "sum"])
        .reset_index(drop=True)
    )
    grouped["slots"] = counts["size"].to_numpy(int)
    grouped["unused_positive_slots"] = counts["sum"].to_numpy(int)
    return grouped


def midnight_daily_audit(data, detail: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    rows = []
    for day, frame in detail[detail["slot"] < 36].groupby("date", sort=True):
        timestamp = pd.Timestamp(day)
        bundle = build_scenarios(data, timestamp, config)
        scenario_totals = bundle.net_load[:, :36].sum(axis=1)
        actual_net = float(
            (frame["actual_load_kwh"] - frame["actual_pv_kwh"]).sum()
        )
        base_net = float((bundle.base_load[:36] - bundle.base_pv[:36]).sum())
        percentile = float(
            (np.sum(scenario_totals < actual_net) + 0.5 * np.sum(scenario_totals == actual_net))
            / len(scenario_totals)
        )
        high_soc = frame["soc_start_kwh"] >= 0.95 * StorageParams().soc_max
        surplus_charge = (
            (frame["charge_kwh"] > TOL)
            & (frame["final_purchase_kwh"] + frame["actual_pv_kwh"] > frame["actual_load_kwh"] + TOL)
        )
        rows.append(
            {
                "date": timestamp,
                "g0_purchase_kwh": float(frame["g0_kwh"].sum()),
                "actual_net_load_kwh": actual_net,
                "actual_load_kwh": float(frame["actual_load_kwh"].sum()),
                "actual_pv_kwh": float(frame["actual_pv_kwh"].sum()),
                "charge_kwh": float(frame["charge_kwh"].sum()),
                "discharge_kwh": float(frame["discharge_kwh"].sum()),
                "unused_kwh": float(frame["curtailment_or_unused_kwh"].sum()),
                "emergency_purchase_kwh": float(frame["emergency_purchase_kwh"].sum()),
                "soc_start_kwh": float(frame["soc_start_kwh"].iloc[0]),
                "soc_end_kwh": float(frame["soc_end_kwh"].iloc[-1]),
                "base_forecast_net_load_kwh": base_net,
                "scenario_net_mean_kwh": float(np.mean(scenario_totals)),
                "scenario_net_median_kwh": float(np.median(scenario_totals)),
                "scenario_net_q80_kwh": float(np.quantile(scenario_totals, 0.8)),
                "actual_net_scenario_percentile": percentile,
                "load_forecast_bias_kwh_forecast_minus_actual": float(
                    bundle.base_load[:36].sum() - frame["actual_load_kwh"].sum()
                ),
                "pv_forecast_bias_kwh_forecast_minus_actual": float(
                    bundle.base_pv[:36].sum() - frame["actual_pv_kwh"].sum()
                ),
                "scenario_residual_mean_bias_kwh": float(np.mean(scenario_totals) - base_net),
                "g0_minus_scenario_mean_kwh": float(frame["g0_kwh"].sum() - np.mean(scenario_totals)),
                "g0_minus_scenario_q80_kwh": float(frame["g0_kwh"].sum() - np.quantile(scenario_totals, 0.8)),
                "surplus_active_charge_kwh": float(frame.loc[surplus_charge, "charge_kwh"].sum()),
                "high_soc_g0_purchase_kwh": float(frame.loc[high_soc, "g0_kwh"].sum()),
                "high_soc_unused_kwh": float(
                    frame.loc[high_soc, "curtailment_or_unused_kwh"].sum()
                ),
                "scenario_count": int(len(scenario_totals)),
                "source_issue_times": "|".join(str(value) for value in bundle.source_issues),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("附件"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_control_compare_full_year/updated_forecast_rolling"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_overpurchase_audit"),
    )
    args = parser.parse_args()
    detail = pd.read_csv(args.run_dir / "problem3_detail.csv")
    daily = pd.read_csv(args.run_dir / "daily_summary.csv")
    checks = json.loads((args.run_dir / "checks.json").read_text(encoding="utf-8"))
    detail["date"] = pd.to_datetime(detail["date"])
    formal = detail[(detail["date"] >= FORMAL_START) & (detail["date"] <= FORMAL_END)].copy()
    if len(formal) != 334 * 144 or not checks["all_checks_passed"]:
        raise ValueError("当前正式结果不完整或约束检查未通过")
    params = StorageParams()
    attributed, metadata = build_detail(formal, params)
    summaries = [
        summarize_dimension(attributed, "time_block", "time_block"),
        summarize_dimension(attributed, "month", "month"),
        summarize_dimension(attributed, "price_band", "price_band"),
        summarize_dimension(attributed, "soc_reaches_upper_bound", "soc_reaches_upper_bound"),
    ]
    source_rows = []
    for source in ("retained_g0", "upward_adjustment", "pv"):
        source_rows.append(
            {
                "dimension": "source_component",
                "dimension_value": source,
                "curtailment_or_unused_kwh": float(
                    attributed[f"{source}_unused_lower_bound_kwh"].sum()
                ),
                f"{source}_unused_lower_bound_kwh": float(
                    attributed[f"{source}_unused_lower_bound_kwh"].sum()
                ),
                f"{source}_unused_upper_bound_kwh": float(
                    attributed[f"{source}_unused_upper_bound_kwh"].sum()
                ),
                "slots": int(len(attributed)),
                "unused_positive_slots": int(
                    (attributed["curtailment_or_unused_kwh"] > TOL).sum()
                ),
            }
        )
    summaries.append(pd.DataFrame(source_rows))
    summary = pd.concat(summaries, ignore_index=True)
    data = read_problem_data(args.data_dir)
    config = RunConfig()
    midnight = midnight_daily_audit(data, attributed, config)
    daily_dates = pd.to_datetime(daily["date"])
    formal_daily = daily[(daily_dates >= FORMAL_START) & (daily_dates <= FORMAL_END)].copy()
    formal_daily["date"] = pd.to_datetime(formal_daily["date"])
    day_attr = attributed.groupby("date")[SUM_COLUMNS].sum().reset_index()
    overpurchase_daily = formal_daily.merge(day_attr, on="date", how="left").merge(
        midnight, on="date", how="left", suffixes=("", "_00_06")
    )
    totals = attributed[SUM_COLUMNS].sum().to_dict()
    midnight_totals = midnight.select_dtypes(include=[np.number]).sum().to_dict()
    audit = {
        "scope": {
            "start": str(FORMAL_START.date()),
            "end": str(FORMAL_END.date()),
            "days": 334,
            "detail_rows": int(len(attributed)),
        },
        "formal_baseline": {
            "total_cost_yuan": float(formal["total_cost_yuan"].sum()),
            "final_contract_purchase_kwh": float(formal["final_purchase_kwh"].sum()),
            "unused_kwh": float(formal["curtailment_or_unused_kwh"].sum()),
        },
        "unused_totals": {key: float(value) for key, value in totals.items()},
        "midnight_00_06_totals": {key: float(value) for key, value in midnight_totals.items()},
        "midnight_00_06_means": {
            key: float(value)
            for key, value in midnight.select_dtypes(include=[np.number]).mean().to_dict().items()
        },
        "attribution_rules": {
            "contract_lower_bound": "max(unused - actual_pv, 0)",
            "contract_upper_bound": "min(unused, final_contract_purchase)",
            "pv_allocation_lower_bound": "max(unused - final_contract_purchase, 0)",
            "pv_allocation_upper_bound": "min(unused, actual_pv)",
            "pv_unavoidable": "max(actual_pv-load-min(power_room,soc_room),0)",
            "joint_ambiguous": "unused-contract_lower_bound-pv_unavoidable",
            "constraint_effects": "Counterfactual and overlapping; do not sum them.",
        },
        **metadata,
        "checks": {
            "source_reconstruction_max_abs_kwh": float(
                attributed["source_attribution_reconstruction_residual_kwh"].abs().max()
            ),
            "original_run_checks_passed": bool(checks["all_checks_passed"]),
            "mature_history_only": bool(checks["mature_history_only"]),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attributed.to_csv(
        args.output_dir / "unused_attribution_detail.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "unused_attribution_summary.csv", index=False, encoding="utf-8-sig"
    )
    overpurchase_daily.to_csv(
        args.output_dir / "overpurchase_daily_summary.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "overpurchase_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
