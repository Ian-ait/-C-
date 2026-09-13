# -*- coding: utf-8 -*-
"""问题4-3：波动电价下的日内滚动购电调整。

代码流程：
1. 读取附件1、附件2、附件3，并额外读取附件4真实电价；
2. 0:00、6:00、12:00、18:00分别预测未来24小时净负荷和电价；
3. 电价预测采用“上周同日同槽基准 + Ridge残差修正”；
4. 日内决策时继续用当天已发生价格残差修正未来短期价格；
5. 将预测电价送入问题三随机MPC求解，真实电价只用于结算；
6. 导出result4-3.xlsx、明细、每日汇总、决策台账和校验结果。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import solve_problem3 as p3
from problem4_price_utils import (
    DEFAULT_RIDGE_ALPHA,
    PriceForecastInfo,
    day_slot_for_target_time,
    actual_price_for_target,
    make_same_week_price_forecast,
    forecast_price_for_targets_ridge_residual,
    read_price_matrix,
)


def read_problem4_data(data_dir: Path):
    """读取问题三基础数据，并增加附件4真实电价矩阵。"""

    data = p3.read_problem_data(data_dir)
    candidates = (data_dir / "附件4(2).xlsx", data_dir / "附件4.xlsx")
    attachment4 = next((path for path in candidates if path.exists()), None)
    if attachment4 is None:
        raise FileNotFoundError(f"缺少附件4：{[str(x) for x in candidates]}")

    price_dates, price_labels, actual_price = read_price_matrix(attachment4)
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    if len(price_dates) != 365 or price_dates.duplicated().any() or not price_dates.equals(expected_dates):
        raise ValueError("附件4必须包含无重复、无缺失且连续的2025年365天")
    if not data.dates.equals(price_dates):
        raise ValueError("附件2日期与附件4日期不一致")
    anchor_checks = {
        "2025-01-01_slot0": bool(np.isclose(actual_price[0, 0], 0.4527)),
        "2025-01-01_slot1": bool(np.isclose(actual_price[0, 1], 0.4733)),
        "2025-12-31_slot143": bool(np.isclose(actual_price[-1, -1], 0.462)),
    }
    if not all(anchor_checks.values()):
        raise ValueError(f"附件4时段映射锚点失败：{anchor_checks}")

    data.actual_price_yuan_per_kwh = actual_price
    data.price_time_labels = price_labels
    data.data_audit["附件4"] = {
        "数据形状": list(actual_price.shape),
        "日期范围": [str(price_dates.min().date()), str(price_dates.max().date())],
        "最小电价": float(np.min(actual_price)),
        "最大电价": float(np.max(actual_price)),
        "均值电价": float(np.mean(actual_price)),
        "用途": "历史电价预测训练、当天已发生价格修正、最终真实结算",
        "时段映射锚点": anchor_checks,
    }
    return data


def _fixed_price_fallback(
    data,
    decision_time: pd.Timestamp,
    targets: pd.DatetimeIndex,
    residual_decay_slots: float,
) -> tuple[np.ndarray, PriceForecastInfo]:
    """1月1日无历史电价时，用附件1固定分时电价作为因果初始化先验。"""

    decision_time = pd.Timestamp(decision_time)
    day_idx = data.day_index(decision_time.date())
    decision_slot = (decision_time.hour * 60 + decision_time.minute) // p3.SLOT_MINUTES
    fixed_profile = np.asarray(data.price, dtype=float)

    intraday_bias = 0.0
    ar1_phi = 0.0
    if decision_slot > 0:
        actual_observed = data.actual_price_yuan_per_kwh[day_idx, :decision_slot]
        residual = actual_observed - fixed_profile[:decision_slot]
        recent = residual[-6:]
        weights = np.arange(1, len(recent) + 1, dtype=float)
        intraday_bias = float(np.average(recent, weights=weights))
        if residual.size >= 2:
            denominator = float(residual[:-1] @ residual[:-1])
            if denominator > 1.0e-12:
                ar1_phi = float(np.clip(
                    (residual[:-1] @ residual[1:]) / denominator,
                    -0.98, 0.98,
                ))

    forecast = np.zeros(len(targets), dtype=float)
    for idx, target in enumerate(targets):
        _, slot = day_slot_for_target_time(pd.Timestamp(target))
        value = float(fixed_profile[slot])
        if decision_slot > 0:
            lead_slots = max(
                int((pd.Timestamp(target) - decision_time).total_seconds() // (p3.SLOT_MINUTES * 60)),
                1,
            )
            value += intraday_bias * (ar1_phi ** lead_slots)
        forecast[idx] = max(value, 0.0)

    return forecast, PriceForecastInfo(
        selected_decay=0.0,
        selection_reason="首日无历史电价，使用附件1固定分时电价初始化",
        intraday_bias=intraday_bias,
        residual_decay_slots=float(residual_decay_slots),
        max_same_week_days=0,
        fallback_days=0,
        selected_alpha=0.0,
        model_name="attachment1_initial",
        latest_actual_feature_time=str(decision_time),
        ar1_phi=ar1_phi,
    )


def forecast_prices_for_decision(
    data,
    decision_time: pd.Timestamp,
    targets: pd.DatetimeIndex,
    args: argparse.Namespace,
) -> tuple[np.ndarray, PriceForecastInfo]:
    """给问题4-3某个决策时刻生成未来目标时刻预测电价。"""

    decision_time = pd.Timestamp(decision_time)
    day_idx = data.day_index(decision_time.date())
    if args.price_mode == "perfect-foresight":
        values = np.asarray([
            actual_price_for_target(data.actual_price_yuan_per_kwh, data.dates, target)
            for target in targets
        ])
        return values, PriceForecastInfo(
            selected_decay=1.0, selection_reason="仅离线下界使用真实未来价格",
            intraday_bias=0.0, residual_decay_slots=0.0,
            max_same_week_days=1, fallback_days=1,
            selected_alpha=DEFAULT_RIDGE_ALPHA, model_name="perfect_foresight",
            latest_actual_feature_time=str(targets[-1]),
        )
    if args.price_model == "lag7":
        lookup = {pd.Timestamp(value).date(): idx for idx, value in enumerate(data.dates)}
        values = []
        for target in targets:
            target_day, slot = day_slot_for_target_time(pd.Timestamp(target))
            idx = lookup[target_day]
            if idx >= 7:
                value = data.actual_price_yuan_per_kwh[idx - 7, slot]
            elif idx > 0:
                value = data.actual_price_yuan_per_kwh[:idx, slot].mean()
            else:
                value = data.price[slot]
            values.append(max(float(value), 0.0))
        return np.asarray(values), PriceForecastInfo(
            selected_decay=1.0, selection_reason="固定上周同刻；不足7天用已有同刻均值",
            intraday_bias=0.0, residual_decay_slots=0.0,
            max_same_week_days=1, fallback_days=7,
            selected_alpha=None, model_name="lag7",
            latest_actual_feature_time=str(decision_time - pd.Timedelta(minutes=10)),
        )
    if day_idx == 0:
        return _fixed_price_fallback(
            data=data,
            decision_time=decision_time,
            targets=targets,
            residual_decay_slots=args.price_residual_decay_slots,
        )

    forecast_decision = (
        decision_time.normalize()
        if args.price_model == "ridge-frozen"
        else decision_time
    )
    result = forecast_price_for_targets_ridge_residual(
        price_matrix=data.actual_price_yuan_per_kwh,
        dates=data.dates,
        decision_time=forecast_decision,
        targets=targets,
        ridge_alpha=DEFAULT_RIDGE_ALPHA,
        candidate_decays=args.price_decay_candidates,
        max_same_week_days=args.max_same_week_price_days,
        fallback_days=args.price_fallback_days,
        min_train_days=1,
        residual_decay_slots=args.price_residual_decay_slots,
    )
    latest = pd.Timestamp(result[1].latest_actual_feature_time)
    if latest > decision_time:
        raise AssertionError(
            f"真实电价特征越过决策时刻：latest={latest}, decision={decision_time}"
        )
    return result


def actual_price_for_day_slot(data, day_idx: int, slot: int) -> float:
    """读取当前日期当前槽位的真实结算电价。"""

    return float(data.actual_price_yuan_per_kwh[day_idx, slot])


def simulate_strategy_q4(
    data,
    start_day: date,
    end_day: date,
    strategy: str,
    initial_soc: float,
    params: p3.StorageParams,
    config: p3.RunConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    """逐日滚动执行问题4-3；预测价决策，真实价结算。"""

    strategy = p3.normalize_strategy(strategy)
    if start_day > end_day:
        raise ValueError("start_day不能晚于end_day")
    if not (params.soc_min <= initial_soc <= params.soc_max):
        raise ValueError("initial_soc超出储能上下界")

    detail_rows: list[dict] = []
    ledger_rows: list[dict] = []
    forecast_audit_rows: list[dict] = []
    daily_rows: list[dict] = []
    current_soc = float(initial_soc)
    started = pd.Timestamp.now()

    day_range = pd.date_range(start_day, end_day, freq="D")
    for completed_days, day_timestamp in enumerate(day_range, start=1):
        day = day_timestamp.date()
        day_idx = data.day_index(day)
        day_soc_start = current_soc
        g0: np.ndarray | None = None
        effective_g: np.ndarray | None = None
        day_detail_start = len(detail_rows)
        decision_slots = p3._decision_slots(strategy)

        for decision_slot in decision_slots:
            decision_time = p3._timestamp_for_slot(day, decision_slot)
            if args.price_mode == "perfect-foresight":
                targets = p3.make_targets(decision_time)
                actual_load = data.actual_vector(targets, "load")
                actual_pv = data.actual_vector(targets, "pv")
                bundle = p3.ScenarioBundle(
                    targets=targets,
                    base_load=actual_load,
                    base_pv=actual_pv,
                    load=actual_load[None, :],
                    pv=actual_pv[None, :],
                    probabilities=np.ones(1),
                    source_issues=[],
                    decision_time=decision_time,
                )
            else:
                bundle = p3.build_scenarios(data, decision_time, config)
            forecast_prices, price_info = forecast_prices_for_decision(
                data=data,
                decision_time=decision_time,
                targets=bundle.targets,
                args=args,
            )

            planning = p3.solve_stochastic_mpc(
                bundle=bundle,
                price=forecast_prices,
                params=params,
                config=config,
                strategy=strategy,
                decision_slot=decision_slot,
                soc_start=current_soc,
                g0_fixed=g0,
            )

            if decision_slot == 0:
                if planning.g0 is None or len(planning.g0) != p3.SLOTS_PER_DAY:
                    raise RuntimeError("0:00求解未返回完整的全天g0")
                g0 = planning.g0.copy()
                effective_g = g0.copy()
            elif g0 is None or effective_g is None:
                raise RuntimeError("日内调整发生在g0建立之前")

            block_end = min(p3.next_update_slot(strategy, decision_slot), p3.SLOTS_PER_DAY)
            block_count = block_end - decision_slot
            before_commit = effective_g.copy()
            effective_g[decision_slot:block_end] = planning.committed_block[:block_count]

            current_day_count = p3.SLOTS_PER_DAY - decision_slot
            proposed = planning.proposed_current_day[:current_day_count]
            for relative_slot in range(current_day_count):
                target_slot = decision_slot + relative_slot
                target_time = p3._timestamp_for_slot(day, target_slot + 1)
                is_locked = target_slot < block_end
                ledger_rows.append(
                    {
                        "strategy": strategy,
                        "price_model": args.price_model,
                        "terminal_value_mode": args.terminal_value_mode,
                        "decision_time": decision_time,
                        "target_time": target_time,
                        "forecast_issue_time": decision_time,
                        "decision_slot": decision_slot,
                        "target_slot": target_slot,
                        "interval": p3.interval_label(target_slot),
                        "forecast_price_yuan_per_kwh": float(forecast_prices[relative_slot]),
                        "g0_kwh": float(g0[target_slot]),
                        "proposed_g_kwh": float(proposed[relative_slot]),
                        "committed_g_kwh": float(effective_g[target_slot]),
                        "previous_committed_g_kwh": float(before_commit[target_slot]),
                        "is_locked_this_decision": bool(is_locked),
                        "scenario_count": int(bundle.net_load.shape[0]),
                        "solver_objective": float(planning.objective),
                    }
                )

            latest_mature_target = (
                max(bundle.source_issues) + pd.Timedelta(hours=24)
                if bundle.source_issues
                else pd.NaT
            )
            actual_price_values = []
            forecast_price_values = []
            for price_offset, target in enumerate(bundle.targets[:current_day_count]):
                target_day, target_slot = day_slot_for_target_time(pd.Timestamp(target))
                if target_day == day:
                    actual_price_values.append(
                        actual_price_for_day_slot(data, day_idx, target_slot)
                    )
                    forecast_price_values.append(float(forecast_prices[price_offset]))
            finite_actual = np.asarray(actual_price_values, dtype=float)
            forecast_compare = np.asarray(forecast_price_values, dtype=float)
            price_mae = (
                float(np.mean(np.abs(forecast_compare - finite_actual)))
                if finite_actual.size
                else np.nan
            )
            forecast_audit_rows.append(
                {
                    "strategy": strategy,
                    "decision_time": decision_time,
                    "forecast_issue_time": decision_time,
                    "scenario_count": int(bundle.net_load.shape[0]),
                    "source_issue_times": "|".join(str(x) for x in bundle.source_issues),
                    "latest_source_target_time": latest_mature_target,
                    "mature_only": bool(
                        pd.isna(latest_mature_target) or latest_mature_target <= decision_time
                    ),
                    "price_model": price_info.model_name,
                    "selected_price_alpha": price_info.selected_alpha,
                    "selected_price_decay": price_info.selected_decay,
                    "price_selection_reason": price_info.selection_reason,
                    "intraday_price_bias_yuan_per_kwh": price_info.intraday_bias,
                    "ar1_phi": price_info.ar1_phi,
                    "latest_actual_price_feature_time": price_info.latest_actual_feature_time,
                    "price_feature_time_ok": bool(
                        args.price_mode == "perfect-foresight"
                        or pd.Timestamp(price_info.latest_actual_feature_time) <= decision_time
                        if price_info.latest_actual_feature_time is not None else True
                    ),
                    "same_day_price_mae_yuan_per_kwh": price_mae,
                    "planned_simultaneous_max_kwh": float(
                        np.minimum(planning.scenario_charge, planning.scenario_discharge).max()
                    ),
                    "solver_status": planning.solver_message,
                }
            )

            actual_net = (
                data.actual_load_kwh[day_idx, decision_slot:block_end]
                - data.actual_pv_kwh[day_idx, decision_slot:block_end]
            )
            is_year_end = day == date(2025, 12, 31)
            if is_year_end:
                feedback_g = proposed.copy()
                scenario_net = bundle.net_load[:, :current_day_count]
                block_forecast_prices = forecast_prices[:current_day_count]
                execution_periods = block_count
                hard_terminal = True
            elif args.terminal_value_mode in ("cross-midnight-dp", "state-dependent"):
                feedback_g = np.average(
                    planning.scenario_g, axis=0, weights=bundle.probabilities
                )
                feedback_g[:block_count] = effective_g[decision_slot:block_end]
                scenario_net = bundle.net_load
                block_forecast_prices = forecast_prices
                execution_periods = block_count
                hard_terminal = False
            else:
                feedback_g = effective_g[decision_slot:block_end]
                scenario_net = bundle.net_load[:, :block_count]
                block_forecast_prices = forecast_prices[:block_count]
                execution_periods = block_count
                hard_terminal = False
            terminal_target = config.final_soc if hard_terminal else None
            feedback = p3.execute_feedback_block(
                committed_g=feedback_g,
                scenario_net=scenario_net,
                probabilities=bundle.probabilities,
                price=block_forecast_prices,
                params=params,
                config=config,
                terminal_target=terminal_target,
                terminal_soc_value=p3._terminal_value(block_forecast_prices, params),
                hard_terminal=hard_terminal,
                actual_net=actual_net,
                soc_start=current_soc,
                execution_periods=execution_periods,
                terminal_base_net=(
                    bundle.base_load[:len(feedback_g)]
                    - bundle.base_pv[:len(feedback_g)]
                ),
            )

            for offset in range(block_count):
                slot = decision_slot + offset
                actual_price = actual_price_for_day_slot(data, day_idx, slot)
                forecast_price = float(forecast_prices[offset])
                planned = float(g0[slot])
                final_g = float(effective_g[slot])
                downward = max(planned - final_g, 0.0)
                upward = max(final_g - planned, 0.0)
                charge = float(feedback["charge"][offset])
                discharge = float(feedback["discharge"][offset])
                emergency = float(feedback["emergency"][offset])
                unused = float(feedback["unused"][offset])
                base_cost = actual_price * planned
                downward_refund = 0.5 * actual_price * downward
                upward_premium = 1.5 * actual_price * upward
                emergency_cost = config.emergency_multiplier * actual_price * emergency
                throughput_cost = config.throughput_penalty * (charge + discharge)
                settlement_cost = base_cost - downward_refund + upward_premium + emergency_cost
                total_cost = settlement_cost + throughput_cost
                target_time = p3._timestamp_for_slot(day, slot + 1)
                load = float(data.actual_load_kwh[day_idx, slot])
                pv = float(data.actual_pv_kwh[day_idx, slot])
                detail_rows.append(
                    {
                        "strategy": strategy,
                        "date": pd.Timestamp(day),
                        "slot": slot,
                        "interval": p3.interval_label(slot),
                        "target_time": target_time,
                        "committing_decision_time": decision_time,
                        "forecast_price_yuan_per_kwh": forecast_price,
                        "actual_price_yuan_per_kwh": actual_price,
                        "price_forecast_error_yuan_per_kwh": forecast_price - actual_price,
                        "lag7_baseline_price_yuan_per_kwh": float(
                            data.actual_price_yuan_per_kwh[day_idx - 7, slot]
                            if day_idx >= 7
                            else (
                                data.actual_price_yuan_per_kwh[:day_idx, slot].mean()
                                if day_idx > 0 else data.price[slot]
                            )
                        ),
                        "price_yuan_per_kwh": actual_price,
                        "g0_kwh": planned,
                        "final_purchase_kwh": final_g,
                        "downward_adjustment_kwh": downward,
                        "upward_adjustment_kwh": upward,
                        "actual_load_kwh": load,
                        "actual_pv_kwh": pv,
                        "charge_kwh": charge,
                        "discharge_kwh": discharge,
                        "emergency_purchase_kwh": emergency,
                        "curtailment_or_unused_kwh": unused,
                        "soc_start_kwh": float(feedback["soc_start"][offset]),
                        "soc_end_kwh": float(feedback["soc_end"][offset]),
                        "boundary_candidate_available": bool(
                            feedback["boundary_candidate_available"][offset]
                        ),
                        "boundary_candidate_used": bool(
                            feedback["boundary_candidate_used"][offset]
                        ),
                        "future_soc_marginal_value_yuan_per_soc_kwh": float(
                            feedback["future_soc_marginal_value"][offset]
                        ),
                        "battery_discharge_marginal_value_yuan_per_output_kwh": float(
                            feedback["battery_discharge_marginal_value"][offset]
                        ),
                        "marginal_value_available": bool(
                            feedback["marginal_value_available"][offset]
                        ),
                        "emergency_unit_price_yuan_per_kwh": (
                            config.emergency_multiplier * actual_price
                        ),
                        "base_purchase_cost_yuan": base_cost,
                        "downward_refund_yuan": downward_refund,
                        "upward_premium_yuan": upward_premium,
                        "emergency_cost_yuan": emergency_cost,
                        "throughput_penalty_yuan": throughput_cost,
                        "settlement_cost_yuan": settlement_cost,
                        "total_cost_yuan": total_cost,
                        "balance_residual_kwh": (
                            final_g + discharge + emergency - charge - unused - load + pv
                        ),
                        "soc_residual_kwh": (
                            float(feedback["soc_end"][offset])
                            - float(feedback["soc_start"][offset])
                            - params.eta_charge * charge
                            + discharge / params.eta_discharge
                        ),
                    }
                )
            current_soc = float(feedback["soc_end"][-1])

        day_frame = pd.DataFrame(detail_rows[day_detail_start:])
        daily_rows.append(
            {
                "strategy": strategy,
                "price_model": args.price_model,
                "terminal_value_mode": args.terminal_value_mode,
                "date": pd.Timestamp(day),
                "soc_start_kwh": day_soc_start,
                "soc_end_kwh": current_soc,
                "base_purchase_cost_yuan": day_frame["base_purchase_cost_yuan"].sum(),
                "downward_refund_yuan": day_frame["downward_refund_yuan"].sum(),
                "upward_premium_yuan": day_frame["upward_premium_yuan"].sum(),
                "adjustment_cost_yuan": (
                    -day_frame["downward_refund_yuan"].sum()
                    + day_frame["upward_premium_yuan"].sum()
                ),
                "emergency_cost_yuan": day_frame["emergency_cost_yuan"].sum(),
                "throughput_penalty_yuan": day_frame["throughput_penalty_yuan"].sum(),
                "total_cost_yuan": day_frame["total_cost_yuan"].sum(),
                "planned_purchase_kwh": day_frame["g0_kwh"].sum(),
                "final_purchase_kwh": day_frame["final_purchase_kwh"].sum(),
                "emergency_purchase_kwh": day_frame["emergency_purchase_kwh"].sum(),
                "curtailment_or_unused_kwh": day_frame["curtailment_or_unused_kwh"].sum(),
                "storage_throughput_kwh": (
                    day_frame["charge_kwh"].sum() + day_frame["discharge_kwh"].sum()
                ),
                "mean_forecast_price_yuan_per_kwh": day_frame[
                    "forecast_price_yuan_per_kwh"
                ].mean(),
                "mean_actual_price_yuan_per_kwh": day_frame["price_yuan_per_kwh"].mean(),
                "price_mae_yuan_per_kwh": (
                    day_frame["forecast_price_yuan_per_kwh"]
                    - day_frame["price_yuan_per_kwh"]
                ).abs().mean(),
            }
        )
        if (
            completed_days % args.progress_every_days == 0
            or completed_days == len(day_range)
        ):
            print(
                f"{completed_days}/{len(day_range)}天 {day}完成 | "
                f"SOC {day_soc_start:.1f}->{current_soc:.1f} | "
                f"紧急购电{daily_rows[-1]['emergency_purchase_kwh']:.1f}kWh | "
                f"总费用{daily_rows[-1]['total_cost_yuan']:.2f}元",
                flush=True,
            )

    runtime = (pd.Timestamp.now() - started).total_seconds()
    return {
        "strategy": strategy,
        "detail": pd.DataFrame(detail_rows),
        "daily": pd.DataFrame(daily_rows),
        "ledger": pd.DataFrame(ledger_rows),
        "forecast_audit": pd.DataFrame(forecast_audit_rows),
        "runtime_seconds": float(runtime),
    }


def save_run_q4(run: dict[str, object], output_dir: Path, checks: dict[str, object]) -> None:
    """保存问题4-3运行明细。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("detail", "problem4_3_detail.csv"),
        ("daily", "daily_summary.csv"),
        ("ledger", "decision_ledger.csv"),
        ("forecast_audit", "price_forecast_audit.csv"),
    ):
        frame = run[key]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    (output_dir / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_q4_3_results(output_dir: Path, comparison: pd.DataFrame, primary_run: dict[str, object] | None) -> None:
    """输出策略比较和主策略每日费用图。"""

    if comparison.empty:
        return

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison["strategy"], comparison["total_cost_yuan"])
    ax.set_title("问题4-3策略总费用对比")
    ax.set_xlabel("策略")
    ax.set_ylabel("费用/元")
    fig.tight_layout()
    fig.savefig(output_dir / "problem4_3_strategy_cost.png", dpi=180)
    plt.close(fig)

    if primary_run is None:
        return
    daily = primary_run["daily"]
    assert isinstance(daily, pd.DataFrame)
    if daily.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    dates = pd.to_datetime(daily["date"])
    ax.plot(dates, daily["total_cost_yuan"], label="总费用")
    ax.plot(dates, daily["emergency_cost_yuan"], label="紧急购电费用")
    ax.set_title("问题4-3主策略每日真实结算费用")
    ax.set_xlabel("日期")
    ax.set_ylabel("费用/元")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "problem4_3_daily_cost.png", dpi=180)
    plt.close(fig)


def summarize_run_q4(
    run: dict[str, object],
    formal_start: date = date(2025, 2, 1),
) -> dict[str, float | str]:
    """按正式评价期汇总问题4-3费用，1月只作为SOC预热期。"""

    detail = run["detail"]
    assert isinstance(detail, pd.DataFrame)
    formal = detail[pd.to_datetime(detail["date"]) >= pd.Timestamp(formal_start)]
    prediction_error = (
        formal["forecast_price_yuan_per_kwh"]
        - formal["actual_price_yuan_per_kwh"]
    )
    lag7_error = (
        formal["lag7_baseline_price_yuan_per_kwh"]
        - formal["actual_price_yuan_per_kwh"]
    )
    actual_high = formal["actual_price_yuan_per_kwh"] >= formal[
        "actual_price_yuan_per_kwh"
    ].quantile(0.8)
    predicted_high = formal["forecast_price_yuan_per_kwh"] >= formal[
        "forecast_price_yuan_per_kwh"
    ].quantile(0.8)
    return {
        "strategy": str(run["strategy"]),
        "price_model": str(formal["price_model"].iloc[0]) if "price_model" in formal else "unknown",
        "terminal_value_mode": str(formal["terminal_value_mode"].iloc[0]) if "terminal_value_mode" in formal else "unknown",
        "formal_start": str(formal_start),
        "formal_days": int(formal["date"].nunique()),
        "total_cost_yuan": float(formal["total_cost_yuan"].sum()),
        "base_purchase_cost_yuan": float(formal["base_purchase_cost_yuan"].sum()),
        "adjustment_cost_yuan": float(
            -formal["downward_refund_yuan"].sum()
            + formal["upward_premium_yuan"].sum()
        ),
        "downward_refund_yuan": float(formal["downward_refund_yuan"].sum()),
        "upward_premium_yuan": float(formal["upward_premium_yuan"].sum()),
        "emergency_cost_yuan": float(formal["emergency_cost_yuan"].sum()),
        "throughput_penalty_yuan": float(formal["throughput_penalty_yuan"].sum()),
        "emergency_purchase_kwh": float(formal["emergency_purchase_kwh"].sum()),
        "upward_adjustment_kwh": float(formal["upward_adjustment_kwh"].sum()),
        "downward_adjustment_kwh": float(formal["downward_adjustment_kwh"].sum()),
        "curtailment_or_unused_kwh": float(
            formal["curtailment_or_unused_kwh"].sum()
        ),
        "storage_throughput_kwh": float(
            formal["charge_kwh"].sum() + formal["discharge_kwh"].sum()
        ),
        "price_mae_yuan_per_kwh": float(
            prediction_error.abs().mean()
        ),
        "price_rmse_yuan_per_kwh": float(np.sqrt(np.mean(prediction_error ** 2))),
        "lag7_mae_yuan_per_kwh": float(lag7_error.abs().mean()),
        "lag7_rmse_yuan_per_kwh": float(np.sqrt(np.mean(lag7_error ** 2))),
        "high_price_top20_recall": float(
            (actual_high & predicted_high).sum() / max(int(actual_high.sum()), 1)
        ),
        "runtime_seconds": float(run["runtime_seconds"]),
    }


def validate_problem4_run(run: dict[str, object], checks: dict[str, object]) -> dict[str, object]:
    """补充检查真实结算、因果电价特征和跨午夜索引。"""
    detail = run["detail"]
    audit = run["forecast_audit"]
    assert isinstance(detail, pd.DataFrame) and isinstance(audit, pd.DataFrame)
    recomputed = (
        detail["base_purchase_cost_yuan"]
        - detail["downward_refund_yuan"]
        + detail["upward_premium_yuan"]
        + detail["emergency_cost_yuan"]
        + detail["throughput_penalty_yuan"]
    )
    cost_residual = float((recomputed - detail["total_cost_yuan"]).abs().max())
    feature_time_ok = bool(audit["price_feature_time_ok"].all())
    price_fields_finite = bool(
        np.isfinite(detail[[
            "forecast_price_yuan_per_kwh", "actual_price_yuan_per_kwh",
            "price_forecast_error_yuan_per_kwh",
        ]].to_numpy(float)).all()
    )
    mapping = {
        "first_slot": day_slot_for_target_time(pd.Timestamp("2025-01-01 00:10"))
        == (date(2025, 1, 1), 0),
        "middle_slot": day_slot_for_target_time(pd.Timestamp("2025-01-01 12:00"))
        == (date(2025, 1, 1), 71),
        "last_slot": day_slot_for_target_time(pd.Timestamp("2025-01-02 00:00"))
        == (date(2025, 1, 1), 143),
        "cross_midnight": day_slot_for_target_time(pd.Timestamp("2025-01-02 00:10"))
        == (date(2025, 1, 2), 0),
    }
    checks.update({
        "no_future_price_feature_leakage": feature_time_ok,
        "forecast_and_actual_price_fields_finite": price_fields_finite,
        "actual_price_used_for_all_settlement_components": cost_residual <= 1e-7,
        "problem4_max_cost_identity_residual_yuan": cost_residual,
        "price_slot_mapping_checks": mapping,
    })
    if not (
        checks["all_checks_passed"] and feature_time_ok and price_fields_finite
        and cost_residual <= 1e-7 and all(mapping.values())
    ):
        raise RuntimeError(f"问题4-3验证失败：{checks}")
    return checks


def write_result4_3(
    template_path: Path, output_path: Path, detail: pd.DataFrame
) -> dict[str, object]:
    """复用问题3填表逻辑，模板源文件保持只读。"""
    return p3.write_result3(template_path, output_path, detail)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题4-3：波动电价日内滚动调整")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("附件"),
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        default=Path("analysis_outputs/problem4_3"),
    )
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--strategy", default="S061218")
    parser.add_argument("--start-date", type=lambda x: pd.Timestamp(x).date(), default=date(2025, 1, 1))
    parser.add_argument("--end-date", type=lambda x: pd.Timestamp(x).date(), default=date(2025, 12, 31))
    parser.add_argument("--max-days", type=int, default=None, help="调试用：从start-date开始只运行N天")
    parser.add_argument("--initial-soc", type=float, default=None)
    parser.add_argument("--ablation", action="store_true", help="运行S0/S06/S0612/S061218")
    parser.add_argument("--all-subsets", action="store_true", help="运行全部8种更新组合并计算Shapley")
    parser.add_argument("--scenarios", type=int, default=p3.RunConfig.scenario_count)
    parser.add_argument("--history-days", type=int, default=p3.RunConfig.history_days)
    parser.add_argument("--seed", type=int, default=p3.RunConfig.seed)
    parser.add_argument("--soc-step", type=float, default=p3.RunConfig.soc_step)
    parser.add_argument("--progress-every-days", type=int, default=1)
    parser.add_argument(
        "--price-mode", choices=("causal", "perfect-foresight"),
        default="causal", help="causal为正式模式；perfect-foresight仅诊断下界",
    )
    parser.add_argument(
        "--price-model",
        choices=("lag7", "ridge-frozen", "ridge-ar1-updated"),
        default="ridge-ar1-updated",
    )
    parser.add_argument(
        "--terminal-value-mode",
        choices=("fixed-linear", "state-dependent"),
        default="fixed-linear",
    )
    parser.add_argument("--compare-price-models", action="store_true")
    parser.add_argument("--compare-terminal-values", action="store_true")
    parser.add_argument(
        "--price-decay-candidates",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.90, 1.00],
    )
    parser.add_argument("--price-lookback-days", type=int, default=35)
    parser.add_argument("--min-price-history-days", type=int, default=7)
    parser.add_argument("--max-same-week-price-days", type=int, default=5)
    parser.add_argument("--price-fallback-days", type=int, default=7)
    parser.add_argument(
        "--price-residual-decay-slots",
        type=float,
        default=36.0,
        help="日内价格残差指数衰减尺度，36个10分钟约为6小时",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scenarios <= 0 or args.history_days <= 0:
        raise ValueError("场景数和历史天数必须为正")
    if args.progress_every_days <= 0:
        raise ValueError("进度打印天数必须为正")
    if args.soc_step <= 0:
        raise ValueError("SOC步长必须为正")
    if args.price_lookback_days <= 0 or args.min_price_history_days <= 0:
        raise ValueError("电价回测窗口和最少历史天数必须为正")
    if args.price_residual_decay_slots <= 0:
        raise ValueError("价格残差衰减尺度必须为正")

    data = read_problem4_data(args.data_dir)
    params = p3.StorageParams()
    initial_soc = params.soc_initial if args.initial_soc is None else args.initial_soc
    config = p3.RunConfig(
        scenario_count=args.scenarios,
        history_days=args.history_days,
        seed=args.seed,
        soc_step=args.soc_step,
        terminal_value_mode=args.terminal_value_mode,
    )
    requested_start = args.start_date
    end_day = args.end_date
    if args.max_days is not None:
        if args.max_days <= 0:
            raise ValueError("--max-days必须为正整数")
        truncated_end = pd.Timestamp(requested_start) + pd.Timedelta(days=args.max_days - 1)
        end_day = min(end_day, truncated_end.date())
    if not date(2025, 1, 1) <= requested_start <= end_day <= date(2025, 12, 31):
        raise ValueError("运行日期必须位于2025年且起止顺序正确")
    start_day = (
        date(2025, 1, 1)
        if requested_start > date(2025, 1, 1) and args.initial_soc is None
        else requested_start
    )

    if args.all_subsets:
        strategies = list(p3.STRATEGIES)
    elif args.ablation:
        strategies = list(p3.CUMULATIVE_ABLATION)
    else:
        strategies = [p3.normalize_strategy(args.strategy)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_mode": "problem4_3_wave_price_mpc",
        "start_day": str(start_day),
        "end_day": str(end_day),
        "initial_soc_kwh": initial_soc,
        "strategies": strategies,
        "storage": asdict(params),
        "config": asdict(config),
        "price_mode": args.price_mode,
        "price_model": {
            "name": "上周同日同槽基准 + Ridge残差修正 + 日内已发生价格残差修正",
            "ridge_alpha": DEFAULT_RIDGE_ALPHA,
            "decay_candidates": args.price_decay_candidates,
            "price_lookback_days": args.price_lookback_days,
            "min_price_history_days": args.min_price_history_days,
            "max_same_week_price_days": args.max_same_week_price_days,
            "price_fallback_days": args.price_fallback_days,
            "price_residual_decay_slots": args.price_residual_decay_slots,
        },
        "data_audit": data.data_audit,
        "time_mapping": "附件4的00:10=slot0，00:00+1=slot143；144列不循环移动",
        "template_time_header_discrepancy": (
            "模板标题可能比强制区间结束时刻口径偏移10分钟；按原始144列顺序写入"
        ),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    price_models = (
        ["lag7", "ridge-frozen", "ridge-ar1-updated"]
        if args.compare_price_models else [args.price_model]
    )
    terminal_modes = (
        ["fixed-linear", "state-dependent"]
        if args.compare_terminal_values else [args.terminal_value_mode]
    )
    original_price_model = args.price_model
    original_terminal_mode = args.terminal_value_mode
    multiple_runs = len(strategies) * len(price_models) * len(terminal_modes) > 1
    summaries = []
    primary_run: dict[str, object] | None = None
    for strategy in strategies:
        for price_model in price_models:
            for terminal_mode in terminal_modes:
                args.price_model = price_model
                args.terminal_value_mode = terminal_mode
                config = p3.RunConfig(
                    scenario_count=args.scenarios,
                    history_days=args.history_days,
                    seed=args.seed,
                    soc_step=args.soc_step,
                    terminal_value_mode=terminal_mode,
                )
                print(
                    f"运行问题4-3 {strategy}/{price_model}/{terminal_mode}: "
                    f"{start_day}至{end_day}", flush=True,
                )
                run = simulate_strategy_q4(
                    data=data, start_day=start_day, end_day=end_day,
                    strategy=strategy, initial_soc=initial_soc,
                    params=params, config=config, args=args,
                )
                checks = validate_problem4_run(
                    run, p3.validate_run(run, params, config)
                )
                strategy_dir = (
                    args.output_dir / f"{strategy}_{price_model}_{terminal_mode}"
                    if multiple_runs else args.output_dir
                )
                save_run_q4(run, strategy_dir, checks)
                summary = summarize_run_q4(run)
                summary.update({
                    "price_model": price_model,
                    "terminal_value_mode": terminal_mode,
                    "price_mode": args.price_mode,
                })
                summaries.append(summary)
                if (
                    strategy == p3.normalize_strategy(args.strategy)
                    and price_model == original_price_model
                    and terminal_mode == original_terminal_mode
                ):
                    primary_run = run

    comparison = pd.DataFrame(summaries)
    comparison_path = args.output_dir / "price_model_cost_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    if args.all_subsets and not multiple_runs:
        shapley = p3.compute_shapley(comparison)
        shapley.to_csv(args.output_dir / "problem4_3_shapley_values.csv", index=False, encoding="utf-8-sig")

    full_year = start_day == date(2025, 1, 1) and end_day == date(2025, 12, 31)
    if full_year and primary_run is not None and args.price_mode == "causal":
        template = args.template or args.data_dir / "附件5" / "result4-3.xlsx"
        workbook_check = write_result4_3(
            template,
            args.output_dir / "result4-3.xlsx",
            primary_run["detail"],
        )
        (args.output_dir / "problem4_3_workbook_check.json").write_text(
            json.dumps(workbook_check, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    args.price_model = original_price_model
    args.terminal_value_mode = original_terminal_mode
    print(comparison.to_string(index=False), flush=True)
    print(f"策略比较：{comparison_path}", flush=True)
    if full_year:
        print(f"提交表格：{args.output_dir / 'result4-3.xlsx'}", flush=True)
    else:
        print("当前不是完整全年运行，已跳过result4-3.xlsx模板导出", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
