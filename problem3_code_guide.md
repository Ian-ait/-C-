# 问题3代码说明

## 1. 项目定位

本项目实现的是一个面向社区负荷、光伏和储能的滚动购电调度器。核心程序是 `solve_problem3.py`，负责：

- 读取附件1、附件2、附件3；
- 在0:00生成全天初始购电计划 `g0`；
- 在可选的6:00、12:00、18:00重新预测并调整未来购电；
- 每10分钟根据真实负荷、真实光伏和当前SOC决定充电、放电、紧急购电；
- 按题目结算规则计算总费用；
- 输出CSV审计文件和题目要求的 `result3.xlsx`。

模型采用“情景抽样 + 线性规划 + 因果反馈DP”的组合方法。它不是严格多阶段随机规划，也不声称给出严格多阶段全局最优解。

## 2. 文件结构

### 主程序

- `solve_problem3.py`：问题3完整调度、回放、验证和输出程序。
- `problem3_net_forecast.py`：可选的直接净负荷梯度提升预测器，对应 `net-hgb` 模式。
- `问题1_单日确定性调度.py`：问题1的LP和DP对照程序。
- `solve_problem2.py`：问题2程序，本项目不修改其输入和输出逻辑。

### 诊断和实验程序

- `audit_problem3_adjustments.py`：分析6/12/18点调整的来源。
- `audit_problem3_overpurchase.py`：分析过量购电和无法利用电量。
- `problem3_cost_diagnostics.py`：费用构成诊断。
- `run_problem3_improvement_tests.py`：候选改进方案测试。
- `run_problem3_multiseed_tests.py`：多随机种子对比测试。
- `run_problem3_net_forecast_tests.py`：`net-hgb`预测测试。
- `check_problem3_net_causality.py`：净负荷预测因果性检查。
- `test_problem3_year_end.py`：年末硬终端回归测试。

### 文档

- `README.md`：项目入口、安装和常用命令。
- `problem3_code_guide.md`：当前代码的结构、数学模型和验证口径。
- `problem3_method.md`：早期方法说明和历史实验记录，部分结果属于修复前版本。

## 3. 输入数据

程序默认从 `附件/` 读取：

- `附件1.xlsx`：144个10分钟时段的电价、参考负荷和光伏预测；
- `附件2.xlsx`：2025全年逐日实际负荷和实际光伏；
- `附件3.xlsx`：每天0:00、6:00、12:00、18:00发布的24小时光伏预测。

程序启动时会检查文件是否存在、工作表结构、行列数、日期范围和数值合法性。

功率统一转换为每10分钟电量：

\[
E_t=P_t\times\frac{10}{60}=\frac{P_t}{6}\quad(\mathrm{kWh}).
\]

内部使用“区间结束时刻”标识交付时段。例如，slot 0对应目标时刻00:10，即区间 `(00:00, 00:10]`；slot 35对应06:00更新边界，6:00更新实际影响从slot 36开始。

## 4. 日内时间尺度

每天有144个10分钟时段：

| 时间 | 作用 |
|---|---|
| 0:00 | 生成全天共享初始购电计划 `g0` |
| 6:00 | 可选的第一次日内更新 |
| 12:00 | 可选的第二次日内更新 |
| 18:00 | 可选的第三次日内更新 |
| 每10分钟 | 真实执行储能反馈和费用结算 |

策略名称表示启用哪些更新：

- `S0`：只有0:00计划；
- `S06`、`S12`、`S18`：单个日内更新；
- `S0612`、`S0618`、`S1218`：两个日内更新；
- `S061218`：启用全部三个更新。

`S6` 是 `S06` 的别名，`Sall` 是 `S061218` 的别名。

日内更新求出的远期计划叫 `proposed_g`。只有从当前更新时间到下一次更新时间之间的交付块写入 `committed_g`，更远的计划不会提前改变合同。这保证历史已锁定的购电计划不会被未来信息改写。

## 5. 预测和情景

### 5.1 光伏预测

附件3提供整点光伏预测。程序以决策时刻已知的实际光伏作为当前锚点，将未来1至24小时预测作为后续锚点，再线性插值到10分钟粒度。

### 5.2 负荷预测

`--load-forecast-mode`支持：

- `current`：历史同槽基准预测；
- `same-week-type`：星期类型修正；
- `trend-weighted`：近期趋势加权；
- `shape-level`：日总量和日形状修正；
- `observed-corrected`：使用当前日期已经观测的前缀误差修正；
- `net-hgb`：使用 `HistGradientBoostingRegressor` 直接预测净负荷。

`net-hgb`需要额外安装 `scikit-learn`。训练样本只使用决策时刻之前已经成熟的数据，不能读取预测目标时刻之后的真实负荷或光伏。

### 5.3 历史轨迹抽样

程序从历史发布记录中筛选成熟的完整24小时轨迹，将负荷残差和光伏残差作为联合轨迹抽样，而不是逐时段独立抽样。这样可以保留：

- 同一天内的时间相关性；
- 负荷和光伏误差的同期关系；
- 不同发布时刻对应的信息边界。

情景概率非负且归一化。历史数据不足时使用冷启动单场景，程序会在审计输出中保留场景数量和来源日期。

## 6. 随机MPC线性规划

`solve_stochastic_mpc`为每个情景建立规划变量。共享的 `g0` 或当前提交块购电量在情景间一致，情景内部的充放电、紧急购电和未来调整用于近似未来补救价值。

主要变量包括：

- `g_t`：最终购电量；
- `c_t`：充电量；
- `d_t`：放电量；
- `S_t`：SOC；
- `e_t`：紧急购电量；
- `w_t`：无法利用的多余电量；
- `a_t^-`、`a_t^+`：相对于0:00 `g0` 的调减和调增量。

每个情景满足：

\[
g_t+d_t+e_t=L_t-V_t+c_t+w_t,
\]

\[
S_{t+1}=S_t+\eta_c c_t-d_t/\eta_d.
\]

当前参数为：

- `SOC_MIN=1200 kWh`；
- `SOC_MAX=10800 kWh`；
- 充放电效率均为0.9；
- 最大功率5000 kW，即每10分钟最大充放电量833.333 kWh；
- 吞吐惩罚为 `1e-4` 元/kWh；
- 紧急购电单价为普通电价的5倍。

规划层默认使用连续变量。若发现同时充放电，程序会记录最大冲突量；现有验证路径要求规划和实际执行均不能出现实际同时充放电。

## 7. 购电结算

最终购电相对于当天0:00的 `g0` 只结算一次：

\[
a_t^- = \max(g_t^0-g_t,0),\qquad
 a_t^+ = \max(g_t-g_t^0,0).
\]

费用为：

\[
C_t=p_tg_t^0-0.5p_ta_t^-+1.5p_ta_t^+
+5p_te_t+\epsilon(c_t+d_t).
\]

因此调减只退回一半、调增按1.5倍收费。该不对称性意味着0点最优计划一般不等于净负荷预测均值，应该由误差分布、未来调整机会、SOC状态和紧急购电风险共同决定。当前代码还没有把固定分位数规则硬编码为 `g0`，相关诊断应通过场景分位数和端到端费用测试完成。

## 8. 真实反馈执行器

情景LP的储能动作不直接复制到现实。真实负荷和光伏到达后，`execute_feedback_block`使用因果DP选择动作，目标是最小化：

\[
\text{当前紧急购电成本}
+\text{吞吐惩罚}
+\text{未来SOC价值}.
\]

### 反馈策略

- `old-target`：旧块末SOC跟踪策略，只用于历史对照；
- `consistent-value`：使用与规划一致的线性SOC剩余价值；
- `grid-boundary`：当前默认策略，允许真实SOC连续变化，并增加精确能量平衡边界动作。

SOC未来价值仍存储在离散网格上，但`grid-boundary`不会把真实SOC强行舍入到网格点。它会额外加入：

- 当前连续SOC的零动作候选；
- 能完全覆盖当前缺口的精确放电候选；
- 能完全吸收当前剩余电量的精确充电候选；
- 对连续SOC使用相邻有限网格点线性插值。

这解决了100 kWh SOC网格无法表达小于90 kWh有效放电量的问题。

## 9. 12月31日硬终端

全年运行从2025-01-01连续回放到2025-12-31，不在2月1日重置SOC。12月31日24:00必须满足：

\[
SOC_{2026-01-01\ 00:00}=6000\ \mathrm{kWh}.
\]

年末反馈DP使用硬终端可达性传播。预测场景阶段只使用物理充放电边界计算可达性，不把真实执行阶段的动作支配规则提前当成预测硬约束。

如果年末真实SOC已经很低，且普通购电不足以同时覆盖负荷和恢复终端SOC，则年末硬终端允许带5倍价格的紧急购电充电。这是专门的终端救援动作，不能与普通日行为混淆：

- 普通日期仍禁止“紧急购电同时充电”；
- 年末救援动作单独计数；
- 仍禁止同时充放电；
- 仍检查功率边界、SOC递推、能量平衡和最终SOC。

这种处理保证硬终端优先满足物理可行性，并把代价显式计入总费用，而不是把不可达错误伪装成正常结果。

## 10. 主要代码函数

- `read_problem_data`：读取和审计输入数据；
- `forecast_load`：生成因果负荷预测；
- `interpolate_pv_forecast`：生成10分钟光伏预测；
- `build_scenarios`：建立成熟历史联合误差情景；
- `solve_stochastic_mpc`：求解随机MPC线性规划；
- `feedback_value_function`：反向计算SOC未来价值；
- `execute_feedback_block`：根据真实值执行因果储能反馈；
- `simulate_strategy`：跨日运行、滚动更新、结算和记录；
- `validate_run`：检查守恒、边界、时序、锁定和年末终端；
- `write_result3`：按模板写出最终工作簿；
- `save_run`：保存CSV、JSON和运行配置。

## 11. 输出文件

每个输出目录通常包括：

- `problem3_detail.csv`：逐10分钟执行明细、SOC和费用；
- `daily_summary.csv`：逐日费用和SOC汇总；
- `decision_ledger.csv`：0/6/12/18点计划、提案和承诺；
- `forecast_audit.csv`：场景来源、成熟性、求解状态；
- `checks.json`：自动验证结果；
- `run_config.json`：完整运行参数和输入审计；
- `emergency_marginal_audit.csv`：紧急购电时的电池边际价值诊断；
- `result3.xlsx`：按题目模板写入的提交工作簿。

## 12. 安装和运行

### 基础依赖

```powershell
uv run --with pandas --with numpy --with scipy --with openpyxl python .\solve_problem3.py --all-year --strategy S061218 --feedback-policy grid-boundary --output-dir .\analysis_outputs\problem3_full_year
```

### 使用净负荷HGB预测

`net-hgb`依赖的安装包名称是 `scikit-learn`，导入模块名称才是 `sklearn`：

```powershell
uv run --with pandas --with numpy --with scipy --with openpyxl --with scikit-learn python .\solve_problem3.py --all-year --strategy S061218 --feedback-policy grid-boundary --load-forecast-mode net-hgb --day-ahead-mode current --terminal-value-mode fixed-linear --seed 2025 --scenarios 12 --history-days 30 --soc-step 100 --progress-every-days 1 --output-dir .\analysis_outputs\problem3_full_year_new
```

先验证依赖：

```powershell
uv run --with scikit-learn python -c "from sklearn.ensemble import HistGradientBoostingRegressor; print('scikit-learn OK')"
```

### 局部验证

```powershell
python -m unittest -v test_problem3_year_end.py
python .\solve_problem3.py --date 2025-12-30 --initial-soc 6000 --strategy S061218 --feedback-policy grid-boundary --forbid-emergency-charging --output-dir .\analysis_outputs\problem3_non_year_end_regression
python .\solve_problem3.py --date 2025-12-31 --initial-soc 10500 --strategy S061218 --feedback-policy grid-boundary --forbid-emergency-charging --load-forecast-mode observed-corrected --day-ahead-mode recourse-aware --output-dir .\analysis_outputs\problem3_year_end_check
```

### 消融测试

```powershell
python .\solve_problem3.py --all-year --all-subsets --output-dir .\analysis_outputs\problem3_ablation
```

## 13. 验证口径

每次运行至少检查：

- 每天144个时段且无重复；
- 所有数量有限且非负；
- 能量平衡残差；
- SOC递推残差和SOC上下界；
- 充放电功率边界；
- 实际无同时充放电；
- 普通日无紧急购电同时充电；
- 年末终端SOC；
- 已锁定合同不被后续更新修改；
- 所有历史场景来源在决策时刻已经成熟；
- 费用分项恒等式。

年末硬终端救援时，`checks.json`会同时报告：

- `emergency_while_charging_count`：全年总数；
- `normal_emergency_while_charging_count`：普通日期总数，应为0；
- `year_end_terminal_rescue_emergency_charging_count`：12月31日终端救援次数；
- `year_end_terminal_residual_kwh`：最终SOC残差，应在容差内。

## 14. 当前限制

1. 随机MPC是两阶段补救近似，不是严格多阶段非预见场景树。
2. 预测场景数量、历史窗口和SOC价值网格会影响结果，需要通过敏感性测试选择参数。
3. `w_t`表示总的无法利用多余电量，不能无条件解释为纯弃光。
4. `net-hgb`需要`scikit-learn`，并且模型训练质量和运行时间需要单独评估。
5. 年末终端救援保证物理可行性，但可能显著增加紧急购电费用；全年总费用仍必须以修复后完整回放为准，不能由局部测试外推。
6. `problem3_method.md`中的修复前全年结果只作为历史对照，不能直接当作当前默认配置的正式结果。

## 15. 推荐工作顺序

1. 先运行 `test_problem3_year_end.py` 和12月30/31日局部回放。
2. 再用最终参数运行全年，并确认 `checks.json` 全部通过。
3. 检查全年紧急购电、终端救援次数和费用分解。
4. 再运行8种更新组合和多随机种子测试。
5. 最后生成并人工抽查 `result3.xlsx`，确认表结构、日期行和时段列没有错位。
