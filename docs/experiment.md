# 实验设置、指标与结果（docs/experiment.md）

> 本文定义重写版毕业设计的实验协议：跑什么、怎么跑、量什么、结果怎么读。
> 表格中的数字来自 **2026-10-20 的正式实验**（`results/default/`，200 轮、seed=0、
> 评估集 100 个电路、CPU），可直接用于论文/答辩；若重跑实验，请以新的
> `results/default/summary.json` 与 `evaluation.json` 为准并更新本文。
> 命令行与 [`docs/architecture.md`](architecture.md) §16 一致。

---

## 1. 实验目的（Research Questions）

| 编号 | 问题 | 对应证据 |
| --- | --- | --- |
| **RQ1** | 训练出的策略能否在保持语义等价的前提下降低电路代价？ | agent 的 `cost_reduction_pct > 0` 且 `equivalence_rate = 1.0` |
| **RQ2** | 学习策略是否优于随机/贪心/不动基线？ | 四种方法在同一评估集上的压缩率对比 |
| **RQ3** | 智能体主要使用哪些规则？行为是否符合设计预期？ | `mean_rule_counts` 分布 |
| **RQ4** | 门数压缩与深度压缩的来源分别是什么？ | `gate_reduction_pct` vs `depth_reduction_pct` |
| **RQ5** | 训练是否稳定、可复现、耗时多少？ | 训练曲线、KL/explained variance、日志与墙钟时间 |
| **RQ6（可选）** | 掩码/势函数奖励等设计是否必要？ | 消融实验（若时间允许；默认不做则标注“未开展”） |

---

## 2. 实验环境

| 项 | 值 |
| --- | --- |
| 操作系统 | Microsoft Windows 11 Pro（10.0.26300） |
| CPU | Intel Core i5-1035G1（4 核 8 线程，1.00 GHz） |
| 内存 | 15.8 GB |
| Python | 3.14.7（项目虚拟环境 `.venv`） |
| 数值库 | numpy 2.5.4 |
| 深度学习框架 | torch 2.14.1+cpu（CPU 版） |
| 绘图 | matplotlib 3.11.2 |
| 可选对照 | cirq 1.7.0（仅用于 `to_cirq` 与模拟器交叉验证，不参与训练） |
| 设备 | `--device cpu`（默认；有 GPU 时可传 `cuda`，本实验不依赖 GPU） |

> 若实验在另一台机器上完成，请把本表替换为实际配置，并在“实验记录”中注明。

---

## 3. 数据集与划分

| 集合 | 规模 | 生成方式 | 种子 | 用途 |
| --- | --- | --- | --- | --- |
| 训练集 | 200 个 5 比特、深度 12 的随机电路 | `random_circuits(count=200, n_qubits=5, depth=12, seed=env.seed)`；`p_two=0.15` | 0 | 训练时在线采样（每个回合 reset 时现生成或从训练池取） |
| 评估集 | 100 个同分布电路 | `scripts/make_dataset.py --count 100 --seed 10000` → `data/eval.json`；若文件不存在，训练器/评估器用 `eval_seed=10000` 现生成 | 10000 | 周期评估与最终对比；**与训练集种子不同，避免泄漏** |

- 每个电路结构相同：5 比特、初始深度 12，每层保证至少一个门；二比特门只放在相邻比特 `(q, q+1)` 上；
- 同一种子下电路完全可复现（`numpy.random.default_rng`）；
- 评估集文件 `data/eval.json` 的 SHA256 为
  `B663DEF1527B0A971E1226BFBD43093AE0173DF37FEB5BDBC3E29363B0DC0465`（冻结快照，可用于核对）；
- 评估集只用于评估，不参与梯度更新；
- **自定义规模/评估集必须显式传 `--dataset`**：`scripts/train.py` 的 `--dataset` 默认是
  `data/eval.json`；若换成 7 比特等自定义规模而未传该参数，Trainer 会读取默认的 5 比特评估集
  并在维度校验时报“比特数不一致”错误。`scripts/evaluate.py` 同样要用相同的 `--dataset`。

---

## 4. 训练配置

### 4.1 环境（`EnvConfig`）

| 参数 | 值 |
| --- | --- |
| `n_qubits` | 5 |
| `n_moments` | 15 |
| `initial_depth` | 12 |
| `p_two` | 0.15 |
| `max_steps` | 48 |
| `lambda_depth` | 0.5 |
| `seed` | 0 |

### 4.2 PPO（`PPOConfig`）

| 参数 | 值 |
| --- | --- |
| `lr` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_epsilon` | 0.2 |
| `value_coef` | 0.5 |
| `entropy_coef` | 0.01 |
| `max_grad_norm` | 0.5 |
| `update_epochs` | 4 |
| `minibatch_size` | 64 |
| `target_kl` | 0.02 |
| `normalize_advantage` | True |

### 4.3 训练器（`TrainConfig`）

| 参数 | 值 |
| --- | --- |
| `iterations` | 200（本次正式实验；`TrainConfig` 默认值为 150） |
| `steps_per_iter`（horizon） | 16 |
| `num_envs` | 8 |
| `hidden_channels` | 64 |
| `eval_every` | 10 |
| `save_every` | 25 |
| `train_circuits` | 200 |
| `eval_circuits` | 100 |
| `eval_seed` | 10000 |
| `out_dir` | `results/default` |
| 评估集路径（`--dataset`） | 默认 `data/eval.json`（本次未显式传；`config.json` 中 `eval_dataset=null`，加载默认路径） |
| `device` | cpu |

### 4.4 基线

| 方法 | 定义 | 作用 |
| --- | --- | --- |
| `noop` | 不执行任何动作 | 零收益下界 |
| `random` | 每步从 `valid_actions` 中均匀随机选 | 探索基线 |
| `greedy` | 每步枚举所有可行动作，选“执行后 `cost` 最小”的动作（并列取最小下标） | 一步贪心强基线 |
| `agent` | 策略网络 `deterministic=True`（argmax） | 被评估的方法 |

四种方法使用**同一评估集、同一环境配置、同一规则集**，唯一区别是动作选择策略。

---

## 5. 指标定义

设评估集有 `N` 个电路，第 `i` 个初始/最终电路为 `C0_i / CT_i`：

| 指标 | 公式 | 单位 | 说明 |
| --- | --- | --- | --- |
| `gate_reduction_pct` | `(1/N) Σ (ops(C0_i) − ops(CT_i)) / ops(C0_i) × 100` | % | 门数压缩率，越高越好 |
| `depth_reduction_pct` | `(1/N) Σ (depth(C0_i) − depth(CT_i)) / depth(C0_i) × 100` | % | 深度压缩率，越高越好 |
| `cost_reduction_pct` | `(1/N) Σ (C(C0_i) − C(CT_i)) / C(C0_i) × 100`，`C = ops + 0.5×depth` | % | **主指标** |
| `equivalence_rate` | `(1/N) Σ [circuits_equivalent(C0_i, CT_i)]` | 比例 | **必须 = 1.0**；否则视为正确性 bug |
| `mean_steps` | `(1/N) Σ steps_i` | 步 | 平均决策步数 |
| `truncation_rate` | `(1/N) Σ [truncated_i]` | 比例 | 因 `max_steps` 截断的回合占比 |
| `mean_rule_counts` | 每种规则在所有回合中的平均使用次数 | 次 | 行为分析 |
| `mean_elapsed_ms` | 每个电路的平均优化耗时 | ms | 效率参考 |

实现位置：`qco/evaluate.py::EvalResult` 与 `evaluate_*` 系列；Markdown 汇总由
`format_comparison(results)` 生成。

> 当前实现聚合的是均值；若论文需要标准差/置信区间，可对评估集多次运行或扩展评估器（未包含在当前实现中，
> 属可选补充）。

---

## 6. 结果表格（正式实验，2026-10-20）

### 6.1 表 1：四种方法主结果（评估集 N=100）

> `agent` 一行为**最优 checkpoint（第 160 轮）**的确定性策略结果；评估集 100 个电路，
> 与训练集（seed=0）不同种子（seed=10000）。数据来源：`results/default/summary.json` 的
> `results`（agent 即 `best_eval`）。

| 方法 | 门数(初始→最终) | 深度(初始→最终) | 门压缩率 | 深度压缩率 | 代价压缩率 | 等价率 | 平均步数 | 截断率 | 平均耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `noop` | 56.33 → 56.33 | 12.00 → 12.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0 ms |
| `random` | 56.33 → 32.84 | 12.00 → 11.44 | 41.70 | 4.67 | 38.14 | 1.00 | 18.29 | 0.00 | 209.21 ms |
| `greedy` | 56.33 → 32.55 | 12.00 → 11.45 | 42.22 | 4.58 | 38.59 | 1.00 | 17.53 | 0.00 | 255.13 ms |
| `agent` | 56.33 → 32.62 | 12.00 → 11.42 | 42.09 | 4.83 | **38.50** | 1.00 | 18.65 | 0.00 | 253.11 ms |

> 说明：
> - 前 8 列（方法、初始→最终门数/深度、三类压缩率、等价率、平均步数）与
>   `format_comparison()` 的 Markdown 输出一致；“截断率/平均耗时”两列取自
>   `EvalResult`/`summary.json`（`format_comparison()` 不显示这两列）。
> - `mean_elapsed_ms` 是**训练期间周期评估**的耗时（共享 CPU）；独立运行
>   `scripts/evaluate.py` 的 `evaluation.json` 中，noop/random/greedy/agent 分别为
>   0.007 / 184.58 / 212.08 / 190.34 ms/电路，无训练负载时更快。
> - 训练到第 200 轮（最后一轮）的 `final_eval`：代价压缩率 38.21%、门压缩率 41.75%、
>   深度压缩率 4.92%、平均步数 18.38、等价率 1.00；best checkpoint 选在第 160 轮，故主表用
>   第 160 轮权重（两者差距 0.3 个百分点以内）。

### 6.2 表 2：规则使用分布（平均每回合调用次数）

| 方法 | `merge_single_qubit` | `cancel_two_qubit` | `move_earlier` | `move_later` |
| --- | --- | --- | --- | --- |
| `noop` | 0.00 | 0.00 | 0.00 | 0.00 |
| `random` | 17.83 | 0.08 | 0.19 | 0.19 |
| `greedy` | 17.08 | 0.08 | 0.25 | 0.12 |
| `agent` | 18.14 | 0.08 | 0.19 | 0.24 |

### 6.3 表 3：训练过程摘要

| 指标 | 值 |
| --- | --- |
| 总迭代数 | 200（默认配置 150） |
| 总环境步数（`iterations × num_envs × steps_per_iter`） | 200 × 8 × 16 = 25,600 |
| 训练总墙钟时间 | 986.17 s（约 16.4 分钟） |
| 每轮平均训练时间 | 约 4.93 s |
| best checkpoint 所在迭代 | 160 |
| best checkpoint 在评估集上的代价 | 38.33（=`best_eval.mean_final_cost`） |
| 第 200 轮 `final_eval` 的代价压缩率 | 38.21% |
| 最终轮训练 `mean_return` | 0.3599 |
| `mean_return` 前 20 轮 / 后 20 轮均值 | 0.3751 / 0.3871（全程范围 0.2735–0.5134） |
| 最终轮训练 `mean_final_cost` | 39.83 |
| 最终 `explained_variance` | 0.8689（最后 50 轮均值 0.838） |
| 最终 `approx_kl` | −0.00106（全程 `abs(approx_kl)` 最大 0.0151，未触发早停） |
| 最终 `entropy` | 1.1518（从 2.214 下降） |
| 网络参数量 | 392,165（`CircuitPolicyNet()` 默认配置） |
| best.pt 大小 | 约 1.58 MB（`state_dict + config + iteration`） |

### 6.4 表 4：复现性检查（同种子重复运行）

> 已做复现性验证：修复版重跑与首跑的 `train_log.csv` / `eval_log.csv` **逐字节相同**
> （SHA256 分别为 `21bd3368ca8c06e4…` 与 `522684e1382db0ac…`）；只有 `config.json` 因新增
> `eval_dataset` 字段而不同，不影响训练/评估结果。

| 项 | 第一次 | 第二次 | 是否一致 |
| --- | --- | --- | --- |
| 最终评估 `cost_reduction_pct`（agent） | 38.50%（第 160 轮 best） | 38.50%（第 160 轮 best） | 一致 |
| best checkpoint 迭代 | 160 | 160 | 一致 |
| `train_log.csv` / `eval_log.csv` | — | — | 逐字节相同（SHA256 一致） |

### 6.5 表 5（可选）：消融实验

> 本次**未开展**（受时间与算力限制）；下表仅保留实验设计，未填入结果，论文中如实写“未开展”。
> 若时间允许，可增加以下消融：

| 变体 | 改动 | `cost_reduction_pct` | 说明 |
| --- | --- | --- | --- |
| full | 掩码 + 势函数奖励（默认） | 38.50（已测，见表 1） | 主方法 |
| no-mask | 用全 1 掩码（非法动作权重置零后重归一化） | 未开展 | 验证掩码必要性 |
| reward = −cost | 每步给 `−C` 而非压缩比例 | 未开展 | 验证奖励设计 |
| greedy 对比 | 一步贪心 | 38.59（已测，见表 1） | 强基线 |

若未做，在论文中明确写“消融实验受时间限制未开展，作为后续工作”。

---

## 7. 图表（已生成并复制到 `docs/figures/`）

> 图由 `scripts/make_figures.py --run results/default --out results/figures` 生成，
> 已复制到 `docs/figures/`（与 `docs/experiment.md` 同目录的相对路径为 `figures/xxx.png`，
> 与根目录 README 的相对路径为 `docs/figures/xxx.png`）。

### 图 1：训练曲线

![训练曲线：回报、评估集代价、门数/深度压缩率](figures/training_curves.png)

内容：2×2 子图——训练回报、评估集代价压缩率、门数压缩率、深度压缩率随迭代变化。
200 轮中训练回报从 0.2735 升到 0.3599，评估集代价压缩率在 38% 上下波动并在第 160 轮达到最优
（38.50%）；门数压缩率约 30% → 42%，深度压缩率整体较低（约 5%）。
数据来自 `results/default/train_log.csv` 与 `eval_log.csv`。

### 图 2：方法对比

![方法对比：noop/random/greedy/agent 压缩率](figures/method_comparison.png)

内容：四种方法在门数/深度/代价压缩率上的分组柱状图；数据来自 `summary.json` 的最终评估。
agent 代价压缩率 38.50%，random 38.14%，greedy 38.59%，noop 0%；等价率均为 1.00。

### 图 3：规则使用占比

![规则使用占比](figures/rule_usage.png)

内容：各方法使用 4 条规则的比例（堆叠柱状图）；用于解释智能体的行为偏好。

### 图 4：优化前后电路对比

![优化前电路](figures/circuit_before.png)

![优化后电路](figures/circuit_after.png)

内容：同一示例电路优化前后的自绘电路图（after 由贪心基线生成）；用于在论文中直观展示
“删了哪些门/压了哪层”。

| 图 | 仓库内引用路径（README/experiment） | 生成来源 |
| --- | --- | --- |
| 图 1 训练曲线 | `docs/figures/training_curves.png`、`figures/training_curves.png` | `scripts/make_figures.py` |
| 图 2 方法对比 | `docs/figures/method_comparison.png`、`figures/method_comparison.png` | 同上 |
| 图 3 规则使用 | `docs/figures/rule_usage.png`、`figures/rule_usage.png` | 同上 |
| 图 4 电路前后 | `docs/figures/circuit_before.png`、`circuit_after.png` | 同上（after 用贪心基线） |

图注（论文可直接使用）：

> 图 1：训练 200 轮中 agent 在训练集上的平均回报（左上）、评估集平均最终代价（右上）、
> 门数压缩率（左下）、深度压缩率（右下）。曲线由 `qco/visualizer.plot_training_curves` 生成，
> 数据来自 `results/default/train_log.csv` 与 `eval_log.csv`；best checkpoint 出现在第 160 轮。

> 图 2：评估集（N=100）上四种方法的压缩率对比。agent 的 `cost_reduction_pct` 为 38.50%，
> random 为 38.14%，greedy 为 38.59%，noop 为 0%；所有方法等价率均为 1.00。

---

## 8. 结果分析（正式实验）

### 8.1 主结果

- `agent`（best checkpoint，第 160 轮）在评估集上的代价压缩率为 **38.50%**，门数压缩率 **42.09%**，
  深度压缩率 **4.83%**；
- 四种方法的等价率均为 **1.00**，说明所有重写规则在真实评估中保持语义（模全局相位）不变，
  没有出现“以改变电路功能换取压缩率”的错误；
- 与 `random`（38.14%）相比，`agent` 的代价压缩率提高 **0.37 个百分点**；
  与一步贪心 `greedy`（38.59%）相比低 **0.09 个百分点**，基本持平；门数压缩率
  （42.09% vs 42.22%）略低，但深度压缩率（4.83% vs 4.58%）略高；
- `noop` 的压缩率为 0，验证指标定义正确；
- 训练到第 200 轮的 `final_eval` 为 38.21%，说明 best checkpoint 选择在第 160 轮是合理的
  （末轮指标没有继续提升）。

### 8.2 门数 vs 深度

门数压缩主要来自 `merge_single_qubit`（每回合平均约 18 次）与少量 `cancel_two_qubit`；
深度压缩主要来自 `move_earlier/move_later`。默认随机电路每层几乎占满所有比特，
`move_*` 要求“门独占一层”，触发机会很少（每回合平均 0.2–0.25 次），因此深度压缩率
（**约 4.8%**）明显低于门数压缩率（**约 42%**）。这不是训练失败，而是**规则适用性**带来的
结构性现象；若要提升深度压缩，需要更稀疏/更长电路，或增加“门交换/重排”类规则（见“局限”）。

### 8.3 行为分析

`agent` 平均使用规则次数为：`merge_single_qubit` **18.14**、`cancel_two_qubit` **0.08**、
`move_earlier` **0.19**、`move_later` **0.24**。与 `greedy` 的分布
（17.08 / 0.08 / 0.25 / 0.12）相比：

- 两者都以 `merge_single_qubit` 为绝对主导，说明“合并相邻单比特门”是主要收益来源；
- `agent` 更倾向于 `move_later`（0.24 vs 0.12），而 `greedy` 更倾向于 `move_earlier`
  （0.25 vs 0.19）；单步收益上两者相当，说明策略学到了与贪心不同的移动偏好但收益接近；
- 两者都几乎不使用 `cancel_two_qubit`（随机电路上相邻同门抵消概率低），符合规则设计预期。

### 8.4 训练稳定性

训练回报从首轮 **0.2735** 逐渐上升：前 20 轮均值 **0.3751**、后 20 轮均值 **0.3871**，
末轮 0.3599，全程范围 0.2735–0.5134；`explained_variance` 从 **−4.15**（首轮价值网络近乎随机）
上升到 **0.87**（最后 50 轮均值 0.838）；`approx_kl` 全程在 ±0.02 以内
（`|approx_kl|` 最大约 **0.0151**，未触发 `target_kl=0.02` 早停）；
`entropy` 从 **2.21** 下降到 **1.15**（最低 1.04），说明策略逐渐确定化但没有完全塌缩。
评估曲线（agent，每 10 轮）：38.30%(it10) → 38.45%(it20) → 38.10%(it100) →
**38.50%(it160，最优)** → 38.21%(it200)；整体训练曲线健康，未出现发散或 NaN。

### 8.5 效率

训练总耗时 **986.17 s（约 16.4 分钟）**，200 轮平均 **4.93 s/轮**（CPU：i5-1035G1）；
独立评估 100 个电路时 agent 平均 **190.34 ms/电路**（约 19.0 s 完成 100 个），
random 为 184.58 ms/电路，greedy 为 212.08 ms/电路；`scripts/evaluate.py` 一次跑完 4 种方法合计
约 **61 秒**。CPU 即可完成全部实验，不依赖 GPU。
（训练期间周期评估因与训练进程共享 CPU，agent 单电路耗时约 253 ms，仅供对比参考。）

### 8.6 结论

- **RQ1 成立**：`agent` 降低代价（38.50%）且等价率 1.00，重写保持语义；
- **RQ2**：`agent`（38.50%）明显优于 `noop`（0%）与 `random`（38.14%），与强基线 `greedy`
  （38.59%）基本持平（低 0.09 个百分点），深度压缩率略高；
- **RQ3/RQ4**：规则分布与门数/深度压缩来源符合设计预期——主收益来自 `merge`，
  `move_*` 受稠密随机电路的适用性限制；
- **RQ5**：固定种子、配置落盘、checkpoint/日志齐全，训练曲线健康；两次重跑的
  `train_log.csv` / `eval_log.csv` 逐字节一致，可复现；
- **RQ6（消融）**：未开展，作为后续工作。

### 8.7 逐电路胜负分析

把 agent 与基线在**每个电路**上比较代价（200 轮训练后的 best.pt）：

| 对比 | agent 更好 | 持平 | agent 更差 | 平均代价差（agent − 基线） |
| --- | --- | --- | --- | --- |
| agent vs `greedy` | 15 | 71 | 14 | +0.055（agent 略高，即略差） |
| agent vs `random` | 19 | 69 | 12 | −0.230（agent 更低，即更好） |
| 门数：agent vs `greedy` | 7 | 85 | 8 | — |
| 深度：agent vs `greedy` | 9 | 83 | 8 | — |

结论口径：**agent 与强贪心基线基本持平**（代价平均仅高 0.055、总压缩率差 0.09 个百分点，
深度压缩率反而高 0.25 个百分点），但**显著优于 random**（平均代价低 0.23）。需要注意：
由于每条合法动作都严格降低代价，`random` 本身已经是很强的下界；性能提升空间主要集中在
规则顺序 / 局部最优选择上。这一结果如实支持“方法正确、学习有效但收益有限”的结论。

---

## 9. 复现命令（与 `docs/architecture.md` §16 一致）

在项目根目录 `C:\Users\zjc20\quantum-circuit-optimization` 下执行（PowerShell）：

```powershell
# 0) 安装依赖（一次性）
C:\Users\zjc20\quantum-circuit-optimization\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1) 生成固定评估集（100 个 5 比特、深度 12 的随机电路）
C:\Users\zjc20\quantum-circuit-optimization\.venv\Scripts\python.exe scripts\make_dataset.py --out data\eval.json --count 100 --n-qubits 5 --depth 12 --seed 10000

# 2) 训练（本次正式实验 200 轮；默认配置 150 轮，可用 150 先跑通流程）
C:\Users\zjc20\quantum-circuit-optimization\.venv\Scripts\python.exe scripts\train.py --out results\default --iterations 200 --num-envs 8 --steps-per-iter 16 --seed 0 --device cpu --max-steps 48 --lambda-depth 0.5

# 3) 评估（best checkpoint + 随机/贪心基线）
C:\Users\zjc20\quantum-circuit-optimization\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint results\default\best.pt --dataset data\eval.json --out results\default\eval --baselines random,greedy

# 4) 生成图表
C:\Users\zjc20\quantum-circuit-optimization\.venv\Scripts\python.exe scripts\make_figures.py --run results\default --out results\figures
```

等价写法（虚拟环境已加入 PATH 时）：

```powershell
python scripts/train.py --out results/default --iterations 200   # 本次实验；默认 150
python scripts/evaluate.py --checkpoint results/default/best.pt --dataset data/eval.json --out results/default/eval --baselines random,greedy
```

> 根目录的 `train.py` / `evaluate.py` 是 `scripts/` 对应脚本的薄封装，参数相同，便于对照原仓库文件名。
> 所有脚本都支持 `--help`；没有 GPU 的环境可直接运行。
>
> **换规模/自定义评估集**：训练脚本必须显式传 `--dataset`（默认 `data/eval.json`），
> 例如 7 比特实验：
>
> ```powershell
> python scripts/make_dataset.py --out data/eval_7q.json --count 100 --n-qubits 7 --depth 16 --seed 10000
> python scripts/train.py --out results/7q --n-qubits 7 --n-moments 20 --initial-depth 16 --dataset data/eval_7q.json
> python scripts/evaluate.py --checkpoint results/7q/best.pt --dataset data/eval_7q.json --out results/7q/eval
> ```
>
> 不传 `--dataset` 时 Trainer 会读取默认的 5 比特评估集，与自定义规模维度不一致并报错。

---

## 10. 产物目录

```
results/default/
├── config.json        # 完整配置（环境/PPO/训练器）
├── train_log.csv      # 每轮训练指标
├── eval_log.csv       # 周期评估指标（最后一轮额外写入 noop/random/greedy）
├── best.pt            # 评估集 mean_final_cost 最优的模型
├── last.pt            # 最后一轮模型
├── checkpoint_0025.pt # 每 save_every 轮保存的周期检查点（默认每 25 轮）
├── ...
├── summary.json       # 最终四方法对比 + 训练耗时 + best 轮次 + comparison_table
└── eval/              # scripts/evaluate.py 输出
    └── evaluation.json

results/figures/
├── training_curves.png
├── method_comparison.png
├── rule_usage.png
├── circuit_before.png
└── circuit_after.png
```

### 10.1 检查点与 `best.pt` 语义

- `best.pt` 保存的是**评估集 `mean_final_cost` 最优那一轮评估时的权重快照**，不是最后一轮权重；
  重新加载 `best.pt` 在评估集上复评，得到的指标应与 `summary.json` 中的 `best_eval` 一致
  （本次实验已验证：两者 agent 指标同为 38.50% / 42.09% / 4.83% / 1.00 / 18.65）；
  该行为由独立审计的回归测试覆盖（`tests/test_audit.py`；审计发现并修复过
  “best.pt 指向错误权重快照”的 P1 问题，详见 [`audit-report.md`](audit-report.md)）。
- `last.pt` 是最后一轮权重；`checkpoint_XXXX.pt` 是每 `save_every` 轮的周期快照；
  三者都包含 `config` 与 `iteration`，可用 `qco.trainer.load_checkpoint` /
  `build_policy_from_checkpoint` 直接重建策略网络。

---

## 11. 实验记录表（每次正式实验填写一行）

| 日期 | 运行目录 | 关键参数（seed / iterations / device） | 代码版本说明 | 主要结果 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | `results/default` | seed=0, iterations=200, num_envs=8, steps_per_iter=16, cpu | 重写版 v1.0.0（best.pt 快照修复后） | agent 38.50% / greedy 38.59% / random 38.14%，等价率 1.00，训练 986.2 s | 主实验 |
| — | — | — | — | — | 同种子复现性重跑未开展（表 4） |

> 建议：正式实验前先做一次 `--iterations 2` 的冒烟测试确认产物齐全，再跑正式轮数（本次 200）；
> 记录机器负载/后台程序，避免墙钟时间对比失真。

---

## 12. 实验注意事项（避免“假结果”）

1. **等价率优先**：任何方法只要等价率 < 1.0，其结果作废；先查规则/模拟器，不要先调参。
2. **训练/评估种子不同**：训练集 seed=0，评估集 seed=10000；不要用训练集当评估集。
3. **不要挑选种子**：主实验用默认 seed=0；若要多次运行，报告均值/方差并说明。
4. **评估用确定性策略**：`evaluate_agent(deterministic=True)`；不要用采样版评估与贪心比较。
5. **截断率要报告**：高截断率说明策略绕路或 `max_steps` 不足，不能只挑“压缩率高”的子集。
6. **baseline 配置一致**：四种方法必须在同一评估集、同一 `EnvConfig`、同一 `lambda_depth` 下运行。
7. **图与表必须对应**：表 1 的主结果与图 2 的柱状图应来自同一次评估输出。
8. **记录耗时口径**：`mean_elapsed_ms` 只包含优化循环（不含训练），机器不同不可直接横向比较。
9. **保留原始产物**：`results/default/`（或带日期的副本）是论文数字的唯一来源，不要手工改 CSV。
10. **消融可选**：未做就在论文中写“未开展”，不要编造。

---

## 13. 结论与讨论（正式实验）

- `agent` 优于 `random`（38.50% vs 38.14%）：说明策略学到了“在可行动作中选更优动作”，
  而不只是随机游走；
- `agent` ≈ `greedy`（38.50% vs 38.59%，低 0.09 个百分点）：说明在 4 条规则的离散动作空间里，
  一步贪心已接近最优；RL 的收益主要体现在“学习到接近贪心的策略”，以及深度压缩率略高
  （4.83% vs 4.58%）；
- 深度压缩率低：解释 `move_*` 的适用条件（门独占层）与随机稠密电路的结构关系；
- 等价率 1.0：这是必须强调的正确性证据，说明“优化”没有以改变计算语义为代价；
- 总体结论：在 4 条规则的小动作空间上，RL 相对强贪心基线的性能提升有限，但本项目完成了
  “正确建模 + 等价保障 + 可训练 + 可评估 + 可复现”的完整闭环，这是原仓库不具备的；
  性能提升（更大规则集、更强网络/训练策略）是后续工作。
