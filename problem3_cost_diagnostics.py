# -*- coding: utf-8 -*-
"""问题3成本诊断：完美信息下界与无法利用电量归因。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from solve_problem3 import StorageParams


FORMAL_START = pd.Timestamp("2025-02-01")
FORMAL_END = pd.Timestamp("2025-12-31")


def _matrix(rows: list[int], cols: list[int], values: list[float], nrow: int, ncol: int):
    return coo_matrix((values, (rows, cols)), shape=(nrow, ncol)).tocsr()


def solve_perfect_information(
    detail: pd.DataFrame,
    params: StorageParams,
    allow_adjustment: bool,
    final_soc: float = 6000.0,
    throughput_penalty: float = 1.0e-4,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """用全年真实净负荷求离线LP；只用于诊断下界。"""

    n = len(detail)
    price = detail["price_yuan_per_kwh"].to_numpy(float)
    net = (
        detail["actual_load_kwh"].to_numpy(float)
        - detail["actual_pv_kwh"].to_numpy(float)
    )
    soc_initial = float(detail["soc_start_kwh"].iloc[0])
    names = ["g0", "g", "charge", "discharge", "soc", "unused", "down", "up"]
    if not allow_adjustment:
        names = ["g", "charge", "discharge", "soc", "unused"]
    offsets = {name: i * n for i, name in enumerate(names)}
    variable_count = len(names) * n

    objective = np.zeros(variable_count)
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * variable_count
    bounds[offsets["charge"]:offsets["charge"] + n] = [
        (0.0, params.max_charge_kwh)
    ] * n
    bounds[offsets["discharge"]:offsets["discharge"] + n] = [
        (0.0, params.max_discharge_kwh)
    ] * n
    bounds[offsets["soc"]:offsets["soc"] + n] = [
        (params.soc_min, params.soc_max)
    ] * n
    objective[offsets["charge"]:offsets["charge"] + n] = throughput_penalty
    objective[offsets["discharge"]:offsets["discharge"] + n] = throughput_penalty
    if allow_adjustment:
        objective[offsets["g0"]:offsets["g0"] + n] = price
        objective[offsets["down"]:offsets["down"] + n] = -0.5 * price
        objective[offsets["up"]:offsets["up"] + n] = 1.5 * price
    else:
        objective[offsets["g"]:offsets["g"] + n] = price

    eq_r: list[int] = []
    eq_c: list[int] = []
    eq_v: list[float] = []
    eq_b: list[float] = []
    ub_r: list[int] = []
    ub_c: list[int] = []
    ub_v: list[float] = []
    ub_b: list[float] = []

    def add_eq(coefficients: list[tuple[int, float]], rhs: float) -> None:
        row = len(eq_b)
        for col, value in coefficients:
            eq_r.append(row); eq_c.append(col); eq_v.append(value)
        eq_b.append(rhs)

    def add_ub(coefficients: list[tuple[int, float]], rhs: float) -> None:
        row = len(ub_b)
        for col, value in coefficients:
            ub_r.append(row); ub_c.append(col); ub_v.append(value)
        ub_b.append(rhs)

    for t in range(n):
        add_eq(
            [
                (offsets["g"] + t, 1.0),
                (offsets["discharge"] + t, 1.0),
                (offsets["charge"] + t, -1.0),
                (offsets["unused"] + t, -1.0),
            ],
            float(net[t]),
        )
        soc_terms = [
            (offsets["soc"] + t, 1.0),
            (offsets["charge"] + t, -params.eta_charge),
            (offsets["discharge"] + t, 1.0 / params.eta_discharge),
        ]
        if t:
            soc_terms.append((offsets["soc"] + t - 1, -1.0))
            rhs = 0.0
        else:
            rhs = soc_initial
        add_eq(soc_terms, rhs)
        if allow_adjustment:
            add_eq(
                [
                    (offsets["g"] + t, 1.0),
                    (offsets["g0"] + t, -1.0),
                    (offsets["down"] + t, 1.0),
                    (offsets["up"] + t, -1.0),
                ],
                0.0,
            )
            add_ub(
                [(offsets["down"] + t, 1.0), (offsets["g0"] + t, -1.0)],
                0.0,
            )
    add_eq([(offsets["soc"] + n - 1, 1.0)], final_soc)

    result = linprog(
        objective,
        A_ub=_matrix(ub_r, ub_c, ub_v, len(ub_b), variable_count) if ub_b else None,
        b_ub=np.asarray(ub_b) if ub_b else None,
        A_eq=_matrix(eq_r, eq_c, eq_v, len(eq_b), variable_count),
        b_eq=np.asarray(eq_b),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"完美信息LP失败 allow_adjustment={allow_adjustment}: "
            f"status={result.status}, message={result.message}"
        )
    x = result.x
    g = x[offsets["g"]:offsets["g"] + n]
    charge = x[offsets["charge"]:offsets["charge"] + n]
    discharge = x[offsets["discharge"]:offsets["discharge"] + n]
    soc = x[offsets["soc"]:offsets["soc"] + n]
    unused = x[offsets["unused"]:offsets["unused"] + n]
    if allow_adjustment:
        g0 = x[offsets["g0"]:offsets["g0"] + n]
        down = x[offsets["down"]:offsets["down"] + n]
        up = x[offsets["up"]:offsets["up"] + n]
        interval_cost = price * g0 - 0.5 * price * down + 1.5 * price * up
    else:
        g0 = g.copy(); down = np.zeros(n); up = np.zeros(n)
        interval_cost = price * g
    throughput_cost = throughput_penalty * (charge + discharge)
    output = detail[["date", "slot", "target_time"]].copy()
    output["mode"] = "adjustment_allowed" if allow_adjustment else "no_adjustment"
    output["perfect_g0_kwh"] = g0
    output["perfect_final_contract_kwh"] = g
    output["perfect_upward_kwh"] = up
    output["perfect_downward_kwh"] = down
    output["perfect_charge_kwh"] = charge
    output["perfect_discharge_kwh"] = discharge
    output["perfect_soc_end_kwh"] = soc
    output["perfect_unused_kwh"] = unused
    output["perfect_contract_cost_yuan"] = interval_cost
    output["perfect_throughput_penalty_yuan"] = throughput_cost
    output["perfect_total_cost_yuan"] = interval_cost + throughput_cost
    summary = {
        "objective_yuan": float(result.fun),
        "total_cost_yuan": float(output["perfect_total_cost_yuan"].sum()),
        "contract_cost_yuan": float(interval_cost.sum()),
        "throughput_penalty_yuan": float(throughput_cost.sum()),
        "unused_kwh": float(unused.sum()),
        "upward_kwh": float(up.sum()),
        "downward_kwh": float(down.sum()),
        "ending_soc_kwh": float(soc[-1]),
        "max_balance_residual_kwh": float(
            np.max(np.abs(g + discharge - charge - unused - net))
        ),
        "max_soc_residual_kwh": float(
            np.max(
                np.abs(
                    soc
                    - np.r_[soc_initial, soc[:-1]]
                    - params.eta_charge * charge
                    + discharge / params.eta_discharge
                )
            )
        ),
    }
    return output, summary


def attribute_unused(detail: pd.DataFrame, params: StorageParams) -> tuple[pd.DataFrame, dict]:
    result = detail.reset_index(drop=True).copy()
    tol = 1.0e-7
    supply_surplus = np.maximum(
        result["final_purchase_kwh"] + result["actual_pv_kwh"]
        - result["actual_load_kwh"],
        0.0,
    )
    pv_surplus = np.maximum(
        result["actual_pv_kwh"] - result["actual_load_kwh"], 0.0
    )
    soc_absorb_cap = np.maximum(
        (params.soc_max - result["soc_start_kwh"]) / params.eta_charge, 0.0
    )
    power_absorb_cap = np.full(len(result), params.max_charge_kwh)
    physical_absorb_cap = np.minimum(soc_absorb_cap, power_absorb_cap)
    actual_unused = result["curtailment_or_unused_kwh"].to_numpy(float)
    unavoidable_pv = np.minimum(
        actual_unused,
        np.maximum(pv_surplus.to_numpy(float) - physical_absorb_cap, 0.0),
    )
    avoidable_contract = np.maximum(actual_unused - unavoidable_pv, 0.0)
    physical_min_unused = np.maximum(
        supply_surplus.to_numpy(float) - physical_absorb_cap, 0.0
    )
    policy_or_value_unused = np.maximum(actual_unused - physical_min_unused, 0.0)
    unused_without_soc_limit = np.maximum(
        supply_surplus.to_numpy(float) - power_absorb_cap, 0.0
    )
    unused_without_power_limit = np.maximum(
        supply_surplus.to_numpy(float) - soc_absorb_cap.to_numpy(float), 0.0
    )
    soc_limit_effect = np.maximum(actual_unused - unused_without_soc_limit, 0.0)
    power_limit_effect = np.maximum(actual_unused - unused_without_power_limit, 0.0)
    soc_binding = (
        (soc_absorb_cap.to_numpy(float) <= power_absorb_cap + tol)
        & (supply_surplus.to_numpy(float) > soc_absorb_cap.to_numpy(float) + tol)
    )
    power_binding = (
        (power_absorb_cap <= soc_absorb_cap.to_numpy(float) + tol)
        & (supply_surplus.to_numpy(float) > power_absorb_cap + tol)
    )
    contract_flag = avoidable_contract > tol
    pv_flag = unavoidable_pv > tol
    categories = []
    cause_text = []
    for index in range(len(result)):
        causes = []
        if pv_flag[index]: causes.append("unavoidable_pv")
        if contract_flag[index]: causes.append("contract_overpurchase")
        if soc_binding[index]: causes.append("soc_high")
        if power_binding[index]: causes.append("charge_power_limit")
        if policy_or_value_unused[index] > tol: causes.append("not_fully_absorbed_by_policy")
        unique = list(dict.fromkeys(causes))
        cause_text.append("|".join(unique) if unique else "none")
        if len(unique) > 1:
            categories.append("multiple_causes")
        elif unique:
            categories.append(unique[0])
        else:
            categories.append("no_unused")
    result["supply_surplus_before_storage_kwh"] = supply_surplus
    result["soc_absorb_capacity_kwh"] = soc_absorb_cap
    result["power_absorb_capacity_kwh"] = power_absorb_cap
    result["physical_absorb_capacity_kwh"] = physical_absorb_cap
    result["unavoidable_pv_unused_kwh"] = unavoidable_pv
    result["avoidable_contract_unused_kwh"] = avoidable_contract
    result["soc_limit_effect_kwh_overlapping"] = soc_limit_effect
    result["power_limit_effect_kwh_overlapping"] = power_limit_effect
    result["policy_or_value_unused_kwh"] = policy_or_value_unused
    result["avoidable_contract_cost_yuan"] = (
        avoidable_contract * result["price_yuan_per_kwh"].to_numpy(float)
    )
    result["unused_primary_category"] = categories
    result["unused_cause_flags"] = cause_text
    positive = result[result["curtailment_or_unused_kwh"] > tol]
    by_category = (
        positive.groupby("unused_primary_category")["curtailment_or_unused_kwh"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "energy_kwh", "count": "slots"})
        .to_dict("index")
    )
    summary = {
        "formal_start": str(FORMAL_START.date()),
        "formal_end": str(FORMAL_END.date()),
        "total_unused_kwh": float(actual_unused.sum()),
        "source_attribution_non_overlapping": {
            "unavoidable_pv_unused_kwh": float(unavoidable_pv.sum()),
            "avoidable_contract_unused_kwh": float(avoidable_contract.sum()),
            "reconstruction_residual_kwh": float(
                actual_unused.sum() - unavoidable_pv.sum() - avoidable_contract.sum()
            ),
            "avoidable_contract_cost_yuan": float(
                result["avoidable_contract_cost_yuan"].sum()
            ),
        },
        "absorption_constraints_overlapping": {
            "soc_limit_effect_kwh": float(soc_limit_effect.sum()),
            "power_limit_effect_kwh": float(power_limit_effect.sum()),
            "policy_or_value_unused_kwh": float(policy_or_value_unused.sum()),
        },
        "primary_category": by_category,
        "definition_note": (
            "Source attribution is non-overlapping. Constraint effects are counterfactual "
            "overlapping diagnostics and must not be summed."
        ),
    }
    return result, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_control_compare_full_year/updated_forecast_rolling"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_adjustment_audit_formal"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_cost_diagnostics"),
    )
    args = parser.parse_args()
    required = [
        args.run_dir / "daily_summary.csv",
        args.run_dir / "problem3_detail.csv",
        args.run_dir / "checks.json",
        args.audit_dir / "adjustment_audit.csv",
        args.audit_dir / "adjustment_audit_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少诊断输入：{missing}")
    daily = pd.read_csv(required[0])
    detail = pd.read_csv(required[1])
    json.loads(required[2].read_text(encoding="utf-8"))
    adjustment_rows = len(pd.read_csv(required[3]))
    adjustment_summary = json.loads(required[4].read_text(encoding="utf-8"))
    detail["date"] = pd.to_datetime(detail["date"])
    formal = detail[(detail["date"] >= FORMAL_START) & (detail["date"] <= FORMAL_END)].copy()
    if len(formal) != 334 * 144 or adjustment_rows != 72144:
        raise ValueError("正式明细或调整审计行数不符合预期")
    if not adjustment_summary["source_path_audit"]["all_sources_mature"]:
        raise ValueError("调整审计发现未成熟历史误差")
    params = StorageParams()
    no_adjust, no_adjust_summary = solve_perfect_information(formal, params, False)
    with_adjust, with_adjust_summary = solve_perfect_information(formal, params, True)
    combined = pd.concat((no_adjust, with_adjust), ignore_index=True)
    current_cost = float(formal["total_cost_yuan"].sum())
    combined["current_formal_total_cost_yuan"] = current_cost
    lower_by_mode = {
        "no_adjustment": no_adjust_summary,
        "adjustment_allowed": with_adjust_summary,
    }
    for mode, values in lower_by_mode.items():
        values["current_cost_gap_yuan"] = current_cost - values["total_cost_yuan"]
        values["current_cost_gap_percent"] = (
            100.0 * (current_cost - values["total_cost_yuan"]) / current_cost
        )
    attributed, unused_summary = attribute_unused(formal, params)
    unused_summary["current_formal_total_cost_yuan"] = current_cost
    unused_summary["perfect_information_lower_bounds"] = lower_by_mode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        args.output_dir / "perfect_information_lower_bound.csv",
        index=False,
        encoding="utf-8-sig",
    )
    attributed.to_csv(
        args.output_dir / "unused_energy_attribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "unused_energy_summary.json").write_text(
        json.dumps(unused_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(unused_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
