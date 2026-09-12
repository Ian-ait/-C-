# -*- coding: utf-8 -*-
"""只读审计问题3滚动调整，不修改优化模型或既有正式结果。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from solve_problem3 import (
    SLOT_MINUTES,
    SLOTS_PER_DAY,
    UPDATE_SLOTS,
    ProblemData,
    RunConfig,
    ScenarioBundle,
    StorageParams,
    _price_for_targets,
    build_scenarios,
    next_update_slot,
    read_problem_data,
    solve_stochastic_mpc,
)


FORMAL_START = pd.Timestamp("2025-02-01")
FORMAL_END = pd.Timestamp("2025-12-31")
UPDATE_HOURS = (6, 12, 18)
STRATEGY = "S061218"


def subset_bundle(bundle: ScenarioBundle, targets: pd.DatetimeIndex) -> ScenarioBundle:
    """按目标时刻抽取连续场景子路径，保留原来源和概率。"""

    positions = bundle.targets.get_indexer(targets)
    if np.any(positions < 0):
        missing = targets[positions < 0]
        raise RuntimeError(f"场景路径缺少目标时刻：{list(missing[:3])}")
    return ScenarioBundle(
        targets=targets,
        base_load=bundle.base_load[positions],
        base_pv=bundle.base_pv[positions],
        load=bundle.load[:, positions],
        pv=bundle.pv[:, positions],
        probabilities=bundle.probabilities.copy(),
        source_issues=list(bundle.source_issues),
        decision_time=bundle.decision_time,
    )


def bundle_with_new_mean(
    old_bundle: ScenarioBundle,
    new_bundle: ScenarioBundle,
) -> ScenarioBundle:
    """更新预测均值但保留0点残差路径及其概率。"""

    load_residual = old_bundle.load - old_bundle.base_load[None, :]
    pv_residual = old_bundle.pv - old_bundle.base_pv[None, :]
    load = np.maximum(new_bundle.base_load[None, :] + load_residual, 0.0)
    pv = np.maximum(new_bundle.base_pv[None, :] + pv_residual, 0.0)
    return ScenarioBundle(
        targets=new_bundle.targets,
        base_load=new_bundle.base_load.copy(),
        base_pv=new_bundle.base_pv.copy(),
        load=load,
        pv=pv,
        probabilities=old_bundle.probabilities.copy(),
        source_issues=list(old_bundle.source_issues),
        decision_time=new_bundle.decision_time,
    )


def source_dates(bundle: ScenarioBundle) -> tuple[str, ...]:
    return tuple(str(pd.Timestamp(value).date()) for value in bundle.source_issues)


def source_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def weighted_quantile(values: np.ndarray, probabilities: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_probabilities = probabilities[order]
    cumulative = np.cumsum(ordered_probabilities)
    return float(ordered_values[np.searchsorted(cumulative, quantile, side="left")])


def scenario_position(value: float, values: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.sum(probabilities[values <= value + 1.0e-9]))


def solve_block(
    bundle: ScenarioBundle,
    data: ProblemData,
    params: StorageParams,
    config: RunConfig,
    decision_slot: int,
    soc_start: float,
    g0: np.ndarray,
) -> np.ndarray:
    planning = solve_stochastic_mpc(
        bundle=bundle,
        price=_price_for_targets(data, bundle.targets),
        params=params,
        config=config,
        strategy=STRATEGY,
        decision_slot=decision_slot,
        soc_start=soc_start,
        g0_fixed=g0,
    )
    return planning.committed_block.copy()


def direction(values: pd.Series, tolerance: float = 1.0e-7) -> pd.Series:
    result = np.zeros(len(values), dtype=int)
    array = values.to_numpy(dtype=float)
    result[array > tolerance] = 1
    result[array < -tolerance] = -1
    return pd.Series(result, index=values.index)


def build_summary(
    audit: pd.DataFrame,
    source_rows: list[dict[str, object]],
    reconstruction: dict[str, float],
    runtime_seconds: float,
) -> dict[str, object]:
    committed = audit[audit["is_committed_this_update"]].copy()
    node_summary: dict[str, dict[str, float | int]] = {}
    for hour, frame in committed.groupby("update_hour"):
        node_summary[str(int(hour))] = {
            "rows": int(len(frame)),
            "upward_adjustment_kwh": float(frame["upward_adjustment_kwh"].sum()),
            "downward_adjustment_kwh": float(frame["downward_adjustment_kwh"].sum()),
            "total_adjustment_kwh": float(frame["absolute_adjustment_kwh"].sum()),
            "adjustment_deviation_cost_yuan": float(
                frame["adjustment_deviation_cost_yuan"].sum()
            ),
            "base_forecast_net_change_kwh_signed": float(
                frame["base_forecast_net_change_kwh"].sum()
            ),
            "base_forecast_net_change_kwh_absolute": float(
                frame["base_forecast_net_change_kwh"].abs().sum()
            ),
            "scenario_expected_net_change_kwh_signed": float(
                frame["scenario_expected_net_change_kwh"].sum()
            ),
            "scenario_expected_net_change_kwh_absolute": float(
                frame["scenario_expected_net_change_kwh"].abs().sum()
            ),
            "mean_soc_deviation_kwh": float(frame["soc_deviation_kwh"].mean()),
            "mean_absolute_soc_deviation_kwh": float(
                frame["soc_deviation_kwh"].abs().mean()
            ),
            "emergency_purchase_after_update_kwh": float(
                frame["emergency_purchase_after_update_kwh"].sum()
            ),
            "unused_after_update_kwh": float(frame["unused_after_update_kwh"].sum()),
        }

    cause_summary: dict[str, dict[str, float]] = {}
    for name in ("forecast_mean", "soc_feedback", "scenario_sample", "other"):
        cause_summary[name] = {
            "signed_adjustment_contribution_kwh": float(
                committed[f"{name}_signed_kwh"].sum()
            ),
            "absolute_adjustment_contribution_kwh": float(
                committed[f"{name}_absolute_kwh"].sum()
            ),
            "adjustment_deviation_cost_contribution_yuan": float(
                committed[f"{name}_cost_yuan"].sum()
            ),
        }

    proposal = audit.copy()
    proposal["proposal_direction"] = direction(proposal["proposed_adjustment_kwh"])
    reversal_targets: set[str] = set()
    reversal_pairs: dict[str, int] = {}
    for target, frame in proposal.groupby("target_time", sort=False):
        frame = frame.sort_values("decision_time")
        nonzero = frame[frame["proposal_direction"] != 0]
        values = nonzero[["update_hour", "proposal_direction"]].to_numpy()
        found = False
        for index in range(1, len(values)):
            if values[index - 1, 1] * values[index, 1] < 0:
                pair = f"{int(values[index - 1, 0]):02d}->{int(values[index, 0]):02d}"
                reversal_pairs[pair] = reversal_pairs.get(pair, 0) + 1
                found = True
        if found:
            reversal_targets.add(str(target))

    source_frame = pd.DataFrame(source_rows)
    source_by_hour = {
        str(int(hour)): {
            "days": int(len(frame)),
            "identical_to_previous_source_set_days": int(
                frame["same_as_previous_sources"].sum()
            ),
            "identical_to_midnight_source_set_days": int(
                frame["same_as_midnight_sources"].sum()
            ),
            "mean_jaccard_vs_previous": float(frame["jaccard_vs_previous"].mean()),
            "mean_jaccard_vs_midnight": float(frame["jaccard_vs_midnight"].mean()),
        }
        for hour, frame in source_frame.groupby("update_hour")
    }
    all_mature = bool(source_frame["all_sources_mature"].all())
    independently_resampled = bool(
        (~source_frame["same_as_previous_sources"]).any()
        or (~source_frame["same_as_midnight_sources"]).any()
    )

    return {
        "scope": {
            "formal_start": str(FORMAL_START.date()),
            "formal_end": str(FORMAL_END.date()),
            "days": int(audit["date"].nunique()),
            "update_rows": int(len(audit)),
            "committed_rows": int(len(committed)),
        },
        "node_summary": node_summary,
        "g0_distribution_summary": {
            "rows": int(len(audit)),
            "mean_g0_minus_scenario_mean_kwh": float(
                (audit["g0_kwh"] - audit["scenario_final_purchase_mean_kwh"]).mean()
            ),
            "mean_g0_minus_scenario_p50_kwh": float(
                (audit["g0_kwh"] - audit["scenario_final_purchase_p50_kwh"]).mean()
            ),
            "share_g0_below_scenario_p50": float(
                (audit["g0_kwh"] < audit["scenario_final_purchase_p50_kwh"] - 1.0e-7).mean()
            ),
            "share_g0_above_scenario_p75": float(
                (audit["g0_kwh"] > audit["scenario_final_purchase_p75_kwh"] + 1.0e-7).mean()
            ),
            "mean_g0_scenario_position": float(audit["g0_scenario_position"].mean()),
            "committed_mean_g0_minus_final_purchase_kwh": float(
                (committed["g0_kwh"] - committed["committed_g_kwh"]).mean()
            ),
            "committed_mean_unused_kwh": float(committed["unused_after_update_kwh"].mean()),
            "committed_mean_emergency_kwh": float(
                committed["emergency_purchase_after_update_kwh"].mean()
            ),
            "interpretation": (
                "scenario_final_purchase is the stochastic LP scenario-specific purchase, "
                "not actual future load. g0_scenario_position is the probability mass at or "
                "below g0 and is a diagnostic, not a replacement rule."
            ),
        },
        "source_path_audit": {
            "independently_resampled_at_updates": independently_resampled,
            "all_sources_mature": all_mature,
            "by_update_hour": source_by_hour,
            "selection_code_evidence": (
                "build_scenarios以decision_time小时筛选历史发布批次，并以"
                "seed + 距YEAR_START的小时数分别初始化随机数生成器"
            ),
        },
        "proposal_reversals": {
            "delivery_targets_with_reverse_direction": len(reversal_targets),
            "adjacent_update_pair_counts": reversal_pairs,
            "note": "同一交付时段只结算一次；此处统计未锁定远期proposed_g的方向反转",
        },
        "cause_decomposition": {
            "order": [
                "midnight_frozen_path_and_original_planned_soc",
                "updated_forecast_mean",
                "updated_scenario_sample",
                "actual_current_soc",
                "full_actual_horizon_and_other_path_effects",
            ],
            "method": (
                "顺序反事实分解；signed贡献与actual_committed_g-g0逐时段闭合，"
                "absolute与cost贡献按相邻反事实的绝对偏差变化计算"
            ),
            "by_cause": cause_summary,
            "maximum_signed_reconstruction_residual_kwh": reconstruction[
                "max_signed_kwh"
            ],
            "maximum_absolute_reconstruction_residual_kwh": reconstruction[
                "max_absolute_kwh"
            ],
            "maximum_cost_reconstruction_residual_yuan": reconstruction[
                "max_cost_yuan"
            ],
        },
        "runtime_seconds": runtime_seconds,
    }


def run_audit(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    data = read_problem_data(args.data_dir)
    params = StorageParams()
    config = RunConfig(
        scenario_count=args.scenarios,
        history_days=args.history_days,
        seed=args.seed,
        soc_step=args.soc_step,
        feedback_policy="grid-boundary",
        forbid_emergency_charging=True,
    )
    detail = pd.read_csv(args.run_dir / "problem3_detail.csv", encoding="utf-8-sig")
    ledger = pd.read_csv(args.run_dir / "decision_ledger.csv", encoding="utf-8-sig")
    for frame in (detail, ledger):
        for column in ("date", "decision_time", "target_time", "committing_decision_time"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column])
    detail = detail[(detail["date"] >= FORMAL_START) & (detail["date"] <= FORMAL_END)]
    ledger = ledger[(ledger["decision_time"] >= FORMAL_START)]

    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    dates = sorted(pd.Timestamp(value).date() for value in detail["date"].unique())
    for day_number, day in enumerate(dates, start=1):
        day_timestamp = pd.Timestamp(day)
        day_detail = detail[detail["date"] == day_timestamp].sort_values("slot")
        day_ledger = ledger[ledger["decision_time"].dt.date == day].copy()
        g0 = day_detail["g0_kwh"].to_numpy(dtype=float)
        midnight_soc = float(day_detail.iloc[0]["soc_start_kwh"])
        bundles = {
            slot: build_scenarios(
                data,
                day_timestamp + pd.Timedelta(minutes=slot * SLOT_MINUTES),
                config,
            )
            for slot in (0, *[UPDATE_SLOTS[hour] for hour in UPDATE_HOURS])
        }
        planning0 = solve_stochastic_mpc(
            bundle=bundles[0],
            price=_price_for_targets(data, bundles[0].targets),
            params=params,
            config=config,
            strategy=STRATEGY,
            decision_slot=0,
            soc_start=midnight_soc,
            g0_fixed=None,
        )
        if planning0.g0 is None or np.max(np.abs(planning0.g0 - g0)) > 1.0e-5:
            raise RuntimeError(f"{day}重建0点g0与正式账本不一致")

        previous_slot = 0
        for hour in UPDATE_HOURS:
            decision_slot = UPDATE_SLOTS[hour]
            block_end = next_update_slot(STRATEGY, decision_slot)
            decision_time = day_timestamp + pd.Timedelta(
                minutes=decision_slot * SLOT_MINUTES
            )
            targets = pd.date_range(
                decision_time + pd.Timedelta(minutes=SLOT_MINUTES),
                periods=SLOTS_PER_DAY - decision_slot,
                freq=f"{SLOT_MINUTES}min",
            )
            block_targets = targets[: block_end - decision_slot]
            midnight_block = subset_bundle(bundles[0], block_targets)
            new_full = bundles[decision_slot]
            new_block = subset_bundle(new_full, block_targets)
            previous_full = bundles[previous_slot]
            previous_remaining = subset_bundle(previous_full, targets)
            new_remaining = subset_bundle(new_full, targets)
            mean_block = bundle_with_new_mean(midnight_block, new_block)

            actual_soc = float(
                day_detail.loc[day_detail["slot"] == decision_slot, "soc_start_kwh"].iloc[0]
            )
            original_planned_soc = float(planning0.mean_soc[decision_slot - 1])
            cf_old = solve_block(
                midnight_block, data, params, config, decision_slot, original_planned_soc, g0
            )
            cf_mean = solve_block(
                mean_block, data, params, config, decision_slot, original_planned_soc, g0
            )
            cf_sample = solve_block(
                new_block, data, params, config, decision_slot, original_planned_soc, g0
            )
            cf_soc = solve_block(
                new_block, data, params, config, decision_slot, actual_soc, g0
            )

            current_ledger = day_ledger[
                day_ledger["decision_time"] == decision_time
            ].sort_values("target_time")
            previous_decision_time = day_timestamp + pd.Timedelta(
                minutes=previous_slot * SLOT_MINUTES
            )
            previous_ledger = day_ledger[
                day_ledger["decision_time"] == previous_decision_time
            ].set_index("target_time")
            if len(current_ledger) != len(targets):
                raise RuntimeError(f"{decision_time}账本目标时段数异常")

            before_base = previous_remaining.base_load - previous_remaining.base_pv
            after_base = new_remaining.base_load - new_remaining.base_pv
            before_expected = np.average(
                previous_remaining.net_load,
                axis=0,
                weights=previous_remaining.probabilities,
            )
            after_expected = np.average(
                new_remaining.net_load,
                axis=0,
                weights=new_remaining.probabilities,
            )
            midnight_sources = source_dates(bundles[0])
            previous_sources = source_dates(previous_full)
            current_sources = source_dates(new_full)
            all_mature = all(
                pd.Timestamp(issue) + pd.Timedelta(hours=24) <= decision_time
                for issue in new_full.source_issues
            )
            source_rows.append(
                {
                    "date": str(day),
                    "update_hour": hour,
                    "same_as_previous_sources": set(current_sources)
                    == set(previous_sources),
                    "same_as_midnight_sources": set(current_sources)
                    == set(midnight_sources),
                    "jaccard_vs_previous": source_jaccard(
                        current_sources, previous_sources
                    ),
                    "jaccard_vs_midnight": source_jaccard(
                        current_sources, midnight_sources
                    ),
                    "all_sources_mature": all_mature,
                }
            )

            actual_block = current_ledger.iloc[: block_end - decision_slot][
                "committed_g_kwh"
            ].to_numpy(dtype=float)
            block_g0 = g0[decision_slot:block_end]
            signed_stages = {
                "forecast_mean": cf_mean - cf_old,
                "scenario_sample": cf_sample - cf_mean,
                "soc_feedback": cf_soc - cf_sample,
            }
            signed_stages["other"] = (
                actual_block - block_g0 - sum(signed_stages.values())
            )
            magnitude_levels = [
                np.zeros_like(block_g0),
                np.abs(cf_old - block_g0),
                np.abs(cf_mean - block_g0),
                np.abs(cf_sample - block_g0),
                np.abs(cf_soc - block_g0),
                np.abs(actual_block - block_g0),
            ]
            magnitude_changes = {
                "forecast_mean": magnitude_levels[2] - magnitude_levels[1],
                "scenario_sample": magnitude_levels[3] - magnitude_levels[2],
                "soc_feedback": magnitude_levels[4] - magnitude_levels[3],
            }
            magnitude_changes["other"] = (
                magnitude_levels[1]
                + magnitude_levels[5]
                - magnitude_levels[4]
            )

            for relative, ledger_row in enumerate(current_ledger.itertuples(index=False)):
                target_time = pd.Timestamp(ledger_row.target_time)
                target_slot = int(ledger_row.target_slot)
                is_committed = target_slot < block_end
                previous_proposed = float(
                    previous_ledger.loc[target_time, "proposed_g_kwh"]
                )
                proposed = float(ledger_row.proposed_g_kwh)
                planned = float(ledger_row.g0_kwh)
                price = float(data.price[target_slot])
                row: dict[str, object] = {
                    "date": day_timestamp,
                    "decision_time": decision_time,
                    "update_hour": hour,
                    "decision_slot": decision_slot,
                    "block_end": block_end,
                    "target_time": target_time,
                    "target_slot": target_slot,
                    "is_committed_this_update": is_committed,
                    "g0_kwh": planned,
                    "scenario_final_purchase_mean_kwh": float(
                        np.average(
                            planning0.scenario_g[:, target_slot],
                            weights=planning0.scenario_g[:, target_slot] * 0.0
                            + bundles[0].probabilities,
                        )
                    ),
                    "scenario_final_purchase_p50_kwh": weighted_quantile(
                        planning0.scenario_g[:, target_slot], bundles[0].probabilities, 0.50
                    ),
                    "scenario_final_purchase_p75_kwh": weighted_quantile(
                        planning0.scenario_g[:, target_slot], bundles[0].probabilities, 0.75
                    ),
                    "scenario_final_purchase_p90_kwh": weighted_quantile(
                        planning0.scenario_g[:, target_slot], bundles[0].probabilities, 0.90
                    ),
                    "g0_scenario_position": scenario_position(
                        planned,
                        planning0.scenario_g[:, target_slot],
                        bundles[0].probabilities,
                    ),
                    "proposed_g_kwh": proposed,
                    "proposed_change_from_previous_kwh": proposed
                    - previous_proposed,
                    "proposed_adjustment_kwh": proposed - planned,
                    "base_forecast_net_before_kwh": float(before_base[relative]),
                    "base_forecast_net_after_kwh": float(after_base[relative]),
                    "base_forecast_net_change_kwh": float(
                        after_base[relative] - before_base[relative]
                    ),
                    "scenario_expected_net_before_kwh": float(
                        before_expected[relative]
                    ),
                    "scenario_expected_net_after_kwh": float(
                        after_expected[relative]
                    ),
                    "scenario_expected_net_change_kwh": float(
                        after_expected[relative] - before_expected[relative]
                    ),
                    "actual_soc_at_update_kwh": actual_soc,
                    "midnight_plan_expected_soc_at_update_kwh": original_planned_soc,
                    "soc_deviation_kwh": actual_soc - original_planned_soc,
                    "previous_source_dates": "|".join(previous_sources),
                    "current_source_dates": "|".join(current_sources),
                    "source_date_jaccard": source_jaccard(
                        current_sources, previous_sources
                    ),
                }
                if is_committed:
                    block_index = target_slot - decision_slot
                    final_g = float(actual_block[block_index])
                    signed_adjustment = final_g - planned
                    detail_row = day_detail.loc[day_detail["slot"] == target_slot].iloc[0]
                    row.update(
                        {
                            "committed_g_kwh": final_g,
                            "signed_adjustment_kwh": signed_adjustment,
                            "upward_adjustment_kwh": max(signed_adjustment, 0.0),
                            "downward_adjustment_kwh": max(-signed_adjustment, 0.0),
                            "absolute_adjustment_kwh": abs(signed_adjustment),
                            "adjustment_deviation_cost_yuan": 0.5
                            * price
                            * abs(signed_adjustment),
                            "emergency_purchase_after_update_kwh": float(
                                detail_row["emergency_purchase_kwh"]
                            ),
                            "unused_after_update_kwh": float(
                                detail_row["curtailment_or_unused_kwh"]
                            ),
                        }
                    )
                    for name in (
                        "forecast_mean",
                        "soc_feedback",
                        "scenario_sample",
                        "other",
                    ):
                        row[f"{name}_signed_kwh"] = float(
                            signed_stages[name][block_index]
                        )
                        row[f"{name}_absolute_kwh"] = float(
                            magnitude_changes[name][block_index]
                        )
                        row[f"{name}_cost_yuan"] = float(
                            0.5 * price * magnitude_changes[name][block_index]
                        )
                rows.append(row)
            previous_slot = decision_slot

        if day_number % args.progress_every_days == 0 or day_number == len(dates):
            elapsed = time.perf_counter() - started
            remaining = elapsed / day_number * (len(dates) - day_number)
            print(
                f"[调整审计] {day_number}/{len(dates)}天 | 当前日期={day} | "
                f"已运行={elapsed:.1f}s | 预计剩余={remaining:.1f}s",
                flush=True,
            )

    audit = pd.DataFrame(rows)
    committed = audit[audit["is_committed_this_update"]]
    signed_rebuilt = sum(
        committed[f"{name}_signed_kwh"]
        for name in ("forecast_mean", "soc_feedback", "scenario_sample", "other")
    )
    absolute_rebuilt = sum(
        committed[f"{name}_absolute_kwh"]
        for name in ("forecast_mean", "soc_feedback", "scenario_sample", "other")
    )
    cost_rebuilt = sum(
        committed[f"{name}_cost_yuan"]
        for name in ("forecast_mean", "soc_feedback", "scenario_sample", "other")
    )
    reconstruction = {
        "max_signed_kwh": float(
            np.max(np.abs(signed_rebuilt - committed["signed_adjustment_kwh"]))
        ),
        "max_absolute_kwh": float(
            np.max(np.abs(absolute_rebuilt - committed["absolute_adjustment_kwh"]))
        ),
        "max_cost_yuan": float(
            np.max(np.abs(cost_rebuilt - committed["adjustment_deviation_cost_yuan"]))
        ),
    }
    summary = build_summary(
        audit,
        source_rows,
        reconstruction,
        time.perf_counter() - started,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(
        args.output_dir / "adjustment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "adjustment_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题3滚动调整只读归因审计")
    parser.add_argument("--data-dir", type=Path, default=Path("附件"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "analysis_outputs/problem3_control_compare_full_year/updated_forecast_rolling"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/problem3_adjustment_audit"),
    )
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--soc-step", type=float, default=100.0)
    parser.add_argument("--progress-every-days", type=int, default=10)
    return parser


if __name__ == "__main__":
    run_audit(build_parser().parse_args())
