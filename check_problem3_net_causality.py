"""Future-data mutation must leave issued forecasts and scenarios identical."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from solve_problem3 import read_problem_data, RunConfig, build_scenarios, YEAR_START
from problem3_net_forecast import get_forecaster

data = read_problem_data(Path('附件'))
results = []
for hour in (0,6,12,18):
    issue = pd.Timestamp('2025-02-01') + pd.Timedelta(hours=hour)
    altered = copy.deepcopy(data)
    if hasattr(altered, '_net_forecaster'):
        del altered._net_forecaster
    first_future = int((issue-YEAR_START).total_seconds()/600)
    altered._actual_load_flat[first_future:] += 100000
    altered._actual_pv_flat[first_future:] += 20000
    altered.actual_load_kwh = altered._actual_load_flat.reshape(365,144).copy()
    altered.actual_pv_kwh = altered._actual_pv_flat.reshape(365,144).copy()
    for key in altered._forecast_lookup:
        if key > issue:
            altered._forecast_lookup[key] += 100000
    original = get_forecaster(data).predict(issue)
    changed = get_forecaster(altered).predict(issue)
    np.testing.assert_array_equal(original, changed)
    cfg = RunConfig(load_forecast_mode='net-hgb')
    a,b = build_scenarios(data,issue,cfg), build_scenarios(altered,issue,cfg)
    np.testing.assert_array_equal(a.net_load,b.net_load)
    current = build_scenarios(data,issue,RunConfig())
    assert a.source_issues == current.source_issues
    results.append(dict(issue=str(issue),future_mutation_max_difference=float(np.max(abs(original-changed))),
        scenario_difference=float(np.max(abs(a.net_load-b.net_load))),same_source_dates=True))
    print(results[-1],flush=True)
out=Path('analysis_outputs/problem3_net_hgb_tests')
out.mkdir(parents=True,exist_ok=True)
(out/'causality_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
