# -*- coding: utf-8 -*-
"""
问题四：仅基于历史电价预测当天144个10分钟电价
方法：7日季节基准 + Ridge残差修正 + 每日扩展窗口重估

约束：
1. 预测某一天 d 时，只使用 d-1 日及以前的价格数据；
2. 不使用负荷、光伏、天气或任何其他外生变量；
3. 日历/时刻变量仅来自时间索引，不包含未来价格；
4. 验证集：2025-07-01 ~ 2025-09-30；
5. 最终独立测试集：2025-10-01 ~ 2025-12-31。
"""

from artifact_tool import Blob, SpreadsheetFile
from datetime import datetime, timedelta
import csv
import math
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_XLSX = "/mnt/data/附件4.xlsx"
PREDICTION_CSV = "/mnt/data/问题四_预测电价_2025-02-01至12-31.csv"


def load_price_matrix(path):
    """用 artifact_tool 读取附件4。返回 dates, prices(365x144)。"""
    wb = SpreadsheetFile.import_xlsx(Blob.load(path))
    sh = wb.worksheets.get_item("Sheet1")
    values = sh.get_range("A1:EO366").values

    excel_serials = np.array([row[0] for row in values[1:]], dtype=float)
    prices = np.array([row[1:] for row in values[1:]], dtype=float)
    dates = [
        datetime(1899, 12, 30) + timedelta(days=float(x))
        for x in excel_serials
    ]

    if prices.shape != (365, 144):
        raise ValueError(f"价格矩阵维度异常：{prices.shape}，预期为 (365, 144)")
    return dates, prices


def build_features(prices, dates, start_lag=14):
    """
    构造只依赖历史价格的特征。
    目标不是直接预测价格，而是预测：
        residual[d,t] = p[d,t] - p[d-7,t]
    最终：
        p_hat[d,t] = p[d-7,t] + residual_hat[d,t]
    """
    X_list = []
    y_list = []
    residual_list = []
    day_list = []
    slot_list = []

    feature_names = [
        "lag1", "lag2", "lag3", "lag7", "lag8", "lag9", "lag14",
        "mean3", "mean7", "mean14", "std7", "std14",
        "res_lag1", "res_lag2", "res_lag7",
        "lag1_neighbor_mean", "lag7_neighbor_mean",
        "prevday_mean", "prevday_std", "prevday_min", "prevday_max",
        "prev7day_mean", "prev7day_std",
        "week_level_diff",
        "slot_sin", "slot_cos",
        "dow_sin", "dow_cos",
        "doy_sin", "doy_cos"
    ]

    n_days, n_slots = prices.shape
    daily_mean = prices.mean(axis=1)
    daily_std = prices.std(axis=1)
    daily_min = prices.min(axis=1)
    daily_max = prices.max(axis=1)

    for d in range(start_lag, n_days):
        dow = dates[d].weekday()
        doy = dates[d].timetuple().tm_yday

        prev7_mean = daily_mean[d-7:d].mean()
        prev7_std = daily_mean[d-7:d].std()
        week_level_diff = daily_mean[d-1] - daily_mean[d-8]

        for s in range(n_slots):
            left = max(0, s-1)
            right = min(n_slots, s+2)

            lag1_neighbor_mean = prices[d-1, left:right].mean()
            lag7_neighbor_mean = prices[d-7, left:right].mean()

            row = [
                prices[d-1, s],
                prices[d-2, s],
                prices[d-3, s],
                prices[d-7, s],
                prices[d-8, s],
                prices[d-9, s],
                prices[d-14, s],

                prices[d-3:d, s].mean(),
                prices[d-7:d, s].mean(),
                prices[d-14:d, s].mean(),
                prices[d-7:d, s].std(),
                prices[d-14:d, s].std(),

                prices[d-1, s] - prices[d-8, s],
                prices[d-2, s] - prices[d-9, s],
                prices[d-7, s] - prices[d-14, s],

                lag1_neighbor_mean,
                lag7_neighbor_mean,

                daily_mean[d-1],
                daily_std[d-1],
                daily_min[d-1],
                daily_max[d-1],

                prev7_mean,
                prev7_std,
                week_level_diff,

                math.sin(2 * math.pi * s / n_slots),
                math.cos(2 * math.pi * s / n_slots),
                math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7),
                math.sin(2 * math.pi * doy / 365),
                math.cos(2 * math.pi * doy / 365),
            ]

            X_list.append(row)
            y_list.append(prices[d, s])
            residual_list.append(prices[d, s] - prices[d-7, s])
            day_list.append(d)
            slot_list.append(s)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    residual = np.asarray(residual_list, dtype=float)
    day_id = np.asarray(day_list, dtype=int)
    slot_id = np.asarray(slot_list, dtype=int)

    return X, y, residual, day_id, slot_id, feature_names


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def day_index(dates, date_text):
    target = datetime.strptime(date_text, "%Y-%m-%d")
    return dates.index(target)


def rolling_daily_ridge(
    X, y, residual, day_id, lag7_col,
    start_day, end_day, alpha
):
    """
    每一天都重新训练一次：
    预测 day=d 时，训练样本严格满足 day_id < d。
    因而不会用到当天或未来的真实价格。
    """
    all_true = []
    all_pred = []
    all_days = []

    for d in range(start_day, end_day + 1):
        train_mask = day_id < d
        test_mask = day_id == d

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        )
        model.fit(X[train_mask], residual[train_mask])

        residual_hat = model.predict(X[test_mask])
        price_hat = X[test_mask, lag7_col] + residual_hat

        all_true.append(y[test_mask])
        all_pred.append(price_hat)
        all_days.append(day_id[test_mask])

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_days),
    )


def curve_diagnostics(y_true, y_pred, day_id):
    corrs = []
    top20_recalls = []

    for d in np.unique(day_id):
        mask = day_id == d
        yt = y_true[mask]
        yp = y_pred[mask]

        if np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
            corrs.append(np.corrcoef(yt, yp)[0, 1])

        k = max(1, int(round(0.2 * len(yt))))
        true_top = set(np.argpartition(yt, -k)[-k:])
        pred_top = set(np.argpartition(yp, -k)[-k:])
        top20_recalls.append(len(true_top & pred_top) / k)

    return {
        "daily_curve_corr": float(np.mean(corrs)),
        "top20_high_price_recall": float(np.mean(top20_recalls)),
    }


def make_interval_labels():
    labels = []
    for i in range(144):
        start_min = i * 10
        end_min = (i + 1) * 10

        sh = start_min // 60
        sm = start_min % 60

        if end_min == 1440:
            end_text = "24:00"
        else:
            eh = end_min // 60
            em = end_min % 60
            end_text = f"{eh:02d}:{em:02d}"

        labels.append(f"{sh:02d}:{sm:02d}-{end_text}")
    return labels


def save_wide_predictions(path, dates, start_day, end_day, pred_flat):
    n_days = end_day - start_day + 1
    pred_matrix = pred_flat.reshape(n_days, 144)
    headers = ["日期"] + make_interval_labels()

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, d in enumerate(range(start_day, end_day + 1)):
            writer.writerow(
                [dates[d].strftime("%Y-%m-%d")]
                + [f"{x:.6f}" for x in pred_matrix[i]]
            )


def main():
    dates, prices = load_price_matrix(INPUT_XLSX)
    X, y, residual, day_id, slot_id, feature_names = build_features(prices, dates)

    lag7_col = feature_names.index("lag7")

    # 1) 验证集：只用于选择 Ridge 正则强度
    val_start = day_index(dates, "2025-07-01")
    val_end = day_index(dates, "2025-09-30")

    candidate_alphas = [0.1, 0.5, 1.0, 10.0]
    val_results = []

    for alpha in candidate_alphas:
        yt, yp, dd = rolling_daily_ridge(
            X, y, residual, day_id, lag7_col,
            val_start, val_end, alpha
        )
        m = calc_metrics(yt, yp)
        val_results.append((alpha, m))
        print(
            f"[Validation] alpha={alpha:<4} "
            f"MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}, R2={m['R2']:.6f}"
        )

    # MAE几乎持平时，优先选RMSE更低且正则更充分的模型。
    best_mae = min(x[1]["MAE"] for x in val_results)
    near_best = [
        x for x in val_results
        if x[1]["MAE"] <= best_mae * 1.0005
    ]
    selected_alpha, selected_val_metrics = min(
        near_best,
        key=lambda z: z[1]["RMSE"]
    )

    print(f"\nSelected alpha = {selected_alpha}")

    # 2) 完全独立测试集：2025-10-01 ~ 2025-12-31
    test_start = day_index(dates, "2025-10-01")
    test_end = day_index(dates, "2025-12-31")

    y_test, p_test, d_test = rolling_daily_ridge(
        X, y, residual, day_id, lag7_col,
        test_start, test_end, selected_alpha
    )
    test_metrics = calc_metrics(y_test, p_test)
    test_diag = curve_diagnostics(y_test, p_test, d_test)

    # 季节性朴素基准：直接使用上周同一天同一时刻价格
    test_mask = (day_id >= test_start) & (day_id <= test_end)
    p_naive = X[test_mask, lag7_col]
    y_naive = y[test_mask]
    d_naive = day_id[test_mask]

    naive_metrics = calc_metrics(y_naive, p_naive)
    naive_diag = curve_diagnostics(y_naive, p_naive, d_naive)

    print("\n=== Independent Holdout: 2025-10-01 ~ 2025-12-31 ===")
    print("Seasonal naive:", naive_metrics)
    print("Ridge residual :", test_metrics)
    print("Ridge diagnostics:", test_diag)

    mae_improve = (
        naive_metrics["MAE"] - test_metrics["MAE"]
    ) / naive_metrics["MAE"]
    rmse_improve = (
        naive_metrics["RMSE"] - test_metrics["RMSE"]
    ) / naive_metrics["RMSE"]

    print(f"MAE improvement : {mae_improve:.2%}")
    print(f"RMSE improvement: {rmse_improve:.2%}")

    # 3) 全部需要预测的时段：2025-02-01 ~ 2025-12-31
    full_start = day_index(dates, "2025-02-01")
    full_end = day_index(dates, "2025-12-31")

    y_full, p_full, d_full = rolling_daily_ridge(
        X, y, residual, day_id, lag7_col,
        full_start, full_end, selected_alpha
    )
    full_metrics = calc_metrics(y_full, p_full)
    full_diag = curve_diagnostics(y_full, p_full, d_full)

    save_wide_predictions(
        PREDICTION_CSV,
        dates,
        full_start,
        full_end,
        p_full
    )

    print("\n=== Full strict rolling backtest: 2025-02-01 ~ 2025-12-31 ===")
    print(full_metrics)
    print(full_diag)
    print(f"\nPrediction file saved to: {PREDICTION_CSV}")


if __name__ == "__main__":
    main()
