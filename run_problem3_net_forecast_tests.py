"""Isolated cross-season same-SOC tests; never writes result3.xlsx."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from solve_problem3 import (RunConfig, StorageParams, read_problem_data,
    simulate_strategy, validate_run, save_run, make_targets, forecast_load,
    interpolate_pv_forecast)
from problem3_net_forecast import get_forecaster
from run_problem3_overpurchase_sensitivity import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[2025])
    parser.add_argument('--months', nargs='+', type=int, default=[2,5,8,11])
    parser.add_argument('--output', default='analysis_outputs/problem3_net_hgb_tests')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    formal = Path('result3.xlsx')
    original_hash = hashlib.sha256(formal.read_bytes()).hexdigest()
    data = read_problem_data(Path('附件'))
    params = StorageParams()
    baseline_detail = pd.read_csv('analysis_outputs/problem3_control_compare_full_year/updated_forecast_rolling/problem3_detail.csv')
    rows, accuracy = [], []
    for month in args.months:
        start = pd.Timestamp(2025, month, 1)
        end = start + pd.offsets.MonthEnd(0)
        first = baseline_detail[pd.to_datetime(baseline_detail.date).eq(start)].iloc[0]
        soc = float(first.soc_start_kwh)
        for seed in args.seeds:
            for mode in ['current', 'net-hgb']:
                folder = out / f'month_{month:02d}' / f'seed_{seed}' / mode
                folder.mkdir(parents=True, exist_ok=True)
                print(f'START month={month} seed={seed} mode={mode}', flush=True)
                if (folder/'checks.json').exists():
                    checks = json.loads((folder/'checks.json').read_text(encoding='utf-8'))
                    run = {'detail':pd.read_csv(folder/'problem3_detail.csv'),
                           'daily':pd.read_csv(folder/'daily_summary.csv')}
                else:
                    run = simulate_strategy(data, start.date(), end.date(), 'S061218', soc,
                        params, RunConfig(seed=seed, load_forecast_mode=mode),
                        progress_every_days=7, control_policy='updated_forecast_rolling')
                    checks = validate_run(run, params, RunConfig(seed=seed, load_forecast_mode=mode))
                    save_run(run, folder, checks)
                if not checks['all_checks_passed']:
                    raise RuntimeError(checks)
                row = summarize(run['detail'], run['daily'], start.date(), end.date())
                row.update(month=month, seed=seed, mode=mode, initial_soc=soc,
                           all_checks_passed=checks['all_checks_passed'])
                rows.append(row)
                pd.DataFrame(rows).to_csv(out/'cost_comparison.csv', index=False)
        for day in pd.date_range(start, end):
            for hour in [0,6,12,18]:
                issue = day + pd.Timedelta(hours=hour)
                tt = make_targets(issue)
                actual = data.actual_vector(tt, 'load') - data.actual_vector(tt, 'pv')
                for mode in ['current','net-hgb']:
                    pred = forecast_load(data, issue, tt, 30, mode)-interpolate_pv_forecast(data,issue,tt)
                    for i, error in enumerate(pred-actual):
                        accuracy.append(dict(month=month,issue_time=str(issue),target_time=str(tt[i]),
                            hour=hour,lead_slot=i+1,lead_block=i//36,mode=mode,error_kwh=float(error)))
        pd.DataFrame(accuracy).to_csv(out/'forecast_errors.csv',index=False)
        pd.DataFrame(get_forecaster(data).audit).to_csv(out/'training_audit.csv',index=False)
    costs=pd.DataFrame(rows)
    base=costs[costs['mode'].eq('current')][['month','seed','total_cost_yuan']].rename(columns={'total_cost_yuan':'baseline_cost'})
    costs=costs.merge(base,on=['month','seed'])
    costs['saving_yuan']=costs.baseline_cost-costs.total_cost_yuan
    costs.to_csv(out/'cost_comparison.csv',index=False)
    costs.groupby(['month','mode']).agg(mean_cost=('total_cost_yuan','mean'),
        mean_saving=('saving_yuan','mean'),worst_saving=('saving_yuan','min'),
        cost_std=('total_cost_yuan','std')).to_csv(out/'cost_summary.csv')
    errors=pd.DataFrame(accuracy)
    metrics=errors.groupby(['month','hour','lead_block','mode']).error_kwh.agg(
        MAE=lambda x:np.abs(x).mean(),RMSE=lambda x:np.sqrt((x*x).mean()),Bias='mean')
    metrics.to_csv(out/'forecast_metrics.csv')
    assert all(x['training_mature'] for x in get_forecaster(data).audit)
    assert hashlib.sha256(formal.read_bytes()).hexdigest()==original_hash
    (out/'metadata.json').write_text(json.dumps(dict(data_audit=data.data_audit,
        seeds=args.seeds,months=args.months,initial_soc_source='formal causal replay',
        scope='independent same-initial-SOC monthly diagnostic, not annual trajectory',
        model='four issue-hour HGBs; lead feature; daily expanding-to-90-day rolling fit',
        initialization='first seven historical issue dates: current causal forecast',
        scenario='historically issued out-of-sample net residual trajectories; same sampled source dates',
        proxy_warning='base_load/load in net-hgb are algebraic compatibility carriers, not load forecasts',
        result3_sha256=original_hash),ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(costs.groupby(['month','mode']).saving_yuan.mean(),flush=True)


if __name__=='__main__':
    main()
