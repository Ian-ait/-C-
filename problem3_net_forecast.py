"""Causal direct net-load forecasting (kWh per ending-labelled 10-minute slot).

Four issue-hour HGB models; lead is an explicit feature. Historical forecasts
are generated with historical training cutoffs, never refitted retrospectively.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


class NetForecaster:
    def __init__(self, data):
        self.data = data
        self.features_cache = {}
        self.predictions = {}
        self.audit = []

    def features(self, issue):
        from solve_problem3 import make_targets, forecast_load, interpolate_pv_forecast
        issue = pd.Timestamp(issue)
        if issue in self.features_cache:
            return self.features_cache[issue]
        targets = make_targets(issue)
        d = self.data
        slots = np.array([d.slot_for_target(t) for t in targets])
        origin_days = (targets - pd.Timedelta(minutes=10)).normalize()
        pv = interpolate_pv_forecast(d, issue, targets)
        baseline = forecast_load(d, issue, targets, 30, 'current') - pv
        columns = [np.arange(1, len(targets)+1), slots,
                   np.sin(slots*2*np.pi/144), np.cos(slots*2*np.pi/144),
                   origin_days.dayofweek.to_numpy(), (origin_days.dayofweek >= 5).astype(float),
                   origin_days.month.to_numpy(), pv, baseline]
        for lag in (1, 2, 7, 14, 28):
            lag_targets = targets - pd.Timedelta(days=lag)
            values = d.actual_vector(lag_targets, 'load') - d.actual_vector(lag_targets, 'pv')
            values[np.asarray(lag_targets > issue)] = np.nan
            columns.append(values)
        prior = np.where(d.dates < issue.normalize())[0]
        for window in (7, 30):
            rows = prior[-window:]
            if len(rows):
                net = d.actual_load_kwh[rows] - d.actual_pv_kwh[rows]
                columns.extend([net.mean(axis=0)[slots], net.std(axis=0)[slots],
                                (net[-1]-net[0])[slots]/max(len(rows)-1, 1)])
            else:
                columns.extend([np.full(len(targets), np.nan)]*3)
        observed = issue.hour*6
        if observed:
            idx = d.day_index(issue.date())
            load = d.actual_load_kwh[idx, :observed]
            solar = d.actual_pv_kwh[idx, :observed]
            stats = [load.mean(), solar.mean(), (load-solar).mean(),
                     load[-18:].mean(), solar[-18:].mean(), (load-solar)[-18:].mean()]
        else:
            stats = [np.nan]*6
        columns.extend([np.full(len(targets), value) for value in stats])
        # Same-hour published-PV bias from fully matured earlier trajectories.
        errors = []
        for offset in range(2, 9):
            old = issue - pd.Timedelta(days=offset)
            if old not in d._forecast_lookup:
                continue
            tt = make_targets(old)
            errors.append(d.actual_vector(tt, 'pv') - interpolate_pv_forecast(d, old, tt))
        columns.append(np.mean(errors, axis=0)[:len(targets)] if errors else np.zeros(len(targets)))
        result = targets, np.column_stack(columns), baseline, pv
        self.features_cache[issue] = result
        return result

    def predict(self, issue):
        issue = pd.Timestamp(issue)
        if issue in self.predictions:
            return self.predictions[issue]
        cutoff = issue.normalize()
        targets, x, baseline, pv = self.features(issue)
        xx, yy, last_targets = [], [], []
        for old_day in pd.date_range(max(pd.Timestamp('2025-01-01'), cutoff-pd.Timedelta(days=90)),
                                     cutoff-pd.Timedelta(days=1)):
            old = old_day + pd.Timedelta(hours=issue.hour)
            tt, xf, _, _ = self.features(old)
            valid = np.asarray(tt <= cutoff)
            y = self.data.actual_vector(tt, 'load') - self.data.actual_vector(tt, 'pv')
            valid &= np.isfinite(y)
            if valid.any():
                xx.append(xf[valid]); yy.append(y[valid]); last_targets.append(tt[valid][-1])
        if len(xx) >= 7:
            model = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=15,
                min_samples_leaf=30, learning_rate=0.08, l2_regularization=1,
                early_stopping=False, random_state=2025)
            with threadpool_limits(limits=1):
                train_x = np.vstack(xx)
                # Midnight observations are entirely missing by definition;
                # drop unavailable/constant columns using training data only.
                keep = np.array([len(np.unique(v[np.isfinite(v)])) > 1
                                 for v in train_x.T])
                model.fit(train_x[:, keep], np.concatenate(yy))
                prediction = model.predict(x[:, keep])
            method = 'hgb'
        else:
            prediction = baseline.copy()
            method = 'causal_initialization'
        self.audit.append(dict(issue_time=str(issue), training_cutoff=str(cutoff),
            max_training_target=str(max(last_targets)) if last_targets else None,
            training_rows=sum(len(v) for v in yy), method=method,
            training_mature=all(t <= cutoff for t in last_targets)))
        self.predictions[issue] = prediction
        return prediction


def get_forecaster(data):
    if not hasattr(data, '_net_forecaster'):
        data._net_forecaster = NetForecaster(data)
    return data._net_forecaster
