# 数模 C 题代码

本仓库包含数模 C 题问题1、问题2和问题3的求解程序。当前最复杂的模块是问题3：它把0点日前购电计划、6/12/18点滚动更新和10分钟储能反馈组合成一个全年连续回放系统。

## 快速入口

- [问题3当前代码说明](problem3_code_guide.md)：推荐首先阅读，内容与当前 `solve_problem3.py` 实现对应。
- [问题3早期方法与实验记录](problem3_method.md)：保留历史建模说明、诊断过程和修复前结果，部分数字不能代表当前默认配置。

## 主要程序

- `solve_problem3.py`：问题3主程序，负责读取数据、预测、场景生成、随机MPC、储能反馈、结算、验证和输出。
- `problem3_net_forecast.py`：可选的直接净负荷HGB预测器，对应 `--load-forecast-mode net-hgb`。
- `问题1_单日确定性调度.py`：问题1的确定性LP/DP对照。
- `solve_problem2.py`：问题2程序。

## 输入数据

默认输入目录为 `附件/`，需要：

- `附件1.xlsx`：价格、参考负荷、光伏预测；
- `附件2.xlsx`：2025全年实际负荷和实际光伏；
- `附件3.xlsx`：0/6/12/18点发布的光伏预测。

程序会检查文件结构、日期范围、数据形状和数值合法性。功率统一换算为10分钟电量，单位为kWh。

## 安装依赖

基础模式：

```powershell
uv run --with pandas --with numpy --with scipy --with openpyxl python .\solve_problem3.py --all-year --strategy S061218 --feedback-policy grid-boundary --output-dir .\analysis_outputs\problem3_full_year
```

使用 `net-hgb` 时，还需要 `scikit-learn`：

```powershell
uv run --with pandas --with numpy --with scipy --with openpyxl --with scikit-learn python .\solve_problem3.py --all-year --strategy S061218 --feedback-policy grid-boundary --load-forecast-mode net-hgb --day-ahead-mode current --terminal-value-mode fixed-linear --seed 2025 --scenarios 12 --history-days 30 --soc-step 100 --output-dir .\analysis_outputs\problem3_full_year_new
```

安装包名称是 `scikit-learn`，代码导入名是 `sklearn`。可用下面的命令验证：

```powershell
uv run --with scikit-learn python -c "from sklearn.ensemble import HistGradientBoostingRegressor; print('scikit-learn OK')"
```

## 常用验证

```powershell
python -m unittest -v test_problem3_year_end.py
python .\solve_problem3.py --date 2025-12-30 --initial-soc 6000 --strategy S061218 --feedback-policy grid-boundary --forbid-emergency-charging --output-dir .\analysis_outputs\problem3_non_year_end_regression
python .\solve_problem3.py --date 2025-12-31 --initial-soc 10500 --strategy S061218 --feedback-policy grid-boundary --forbid-emergency-charging --load-forecast-mode observed-corrected --day-ahead-mode recourse-aware --output-dir .\analysis_outputs\problem3_year_end_check
```

## 问题3的核心逻辑

1. 0:00使用当前预测生成全天初始购电计划 `g0`。
2. 6:00、12:00、18:00按策略重算未来区间；已锁定交付块不再修改。
3. 历史成熟误差以完整24小时联合轨迹抽样，输入随机MPC。
4. 真实负荷和光伏到达后，由因果DP决定充电、放电、紧急购电和无法利用电量。
5. 全年SOC连续传递，12月31日24:00必须回到6000 kWh。
6. 输出明细、账本、预测审计、检查结果和模板工作簿。

当前默认反馈策略是 `grid-boundary`：未来价值仍使用离散SOC网格，但真实SOC和精确平衡动作不再被强制舍入。年末硬终端允许有明确成本的终端救援紧急充电，普通日期仍严格禁止紧急购电同时充电。

## 输出文件

每个运行目录通常包含：

- `problem3_detail.csv`：逐10分钟动作、SOC和费用；
- `daily_summary.csv`：逐日汇总；
- `decision_ledger.csv`：计划、提案和已承诺购电；
- `forecast_audit.csv`：预测来源、成熟性和求解状态；
- `checks.json`：自动验证结果；
- `run_config.json`：运行配置和输入审计；
- `result3.xlsx`：按题目模板生成的提交文件。

详细的函数说明、变量、公式、策略差异和限制请阅读 [problem3_code_guide.md](problem3_code_guide.md)。

## 当前结果口径

`problem3_method.md`中标记为“修复前”的全年费用、消融和Shapley结果是历史实验记录。它们不能直接作为修复后 `grid-boundary` 或 `net-hgb` 配置的正式全年结果。正式结论必须来自当前代码的完整全年回放，并且 `checks.json` 中的 `all_checks_passed` 必须为 `true`。


## 目录

- `问题1_单日确定性调度.py`：问题1 单日确定性调度，含 LP 基准模型 + DP 对比模型

## 运行环境

- Python 3.13+
- 依赖：`pandas`、`numpy`、`scipy`、`openpyxl`

安装依赖：

```bash
pip install pandas numpy scipy openpyxl
```

## 运行

```bash
python 问题1_单日确定性调度.py
```

## 说明

- 数据文件 `附件1.xlsx` 与输出目录 `analysis_outputs/` 未上传，请自行放置在 `DATA_PATH` / `OUT_DIR` 指定的路径。
- LP 为连续变量全局最优基准模型；DP 用 SOC 离散化做对比验证，离散步长越小越接近 LP。
