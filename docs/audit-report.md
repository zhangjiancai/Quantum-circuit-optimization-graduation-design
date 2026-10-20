# 独立审计报告（T4）

> 审计员：T4（独立于 qco / scripts / tests 的实现者）
> 审计对象：`C:\Users\zjc20\quantum-circuit-optimization`（重写版毕业设计工程，**冻结快照**）
> 契约基准：[`docs/architecture.md`](architecture.md)（唯一接口契约）
> 审计测试：[`../tests/test_audit.py`](../tests/test_audit.py)（55 个独立用例，2599 行）
> 审计方式：**不信任现有测试与实现**；关键性质全部用本文件内独立实现的参考
> （独立 11 门矩阵、独立 kron/置换嵌入的幺正矩阵、独立逐环境 GAE 递推环、
> 独立 masked log-softmax、独立动作布局换算）与对抗性构造交叉验证；
> 发现 P1 即时上报 Lead 并附最小复现，修复后转化为常驻回归测试。

---

## 0. 结论摘要

**最终判定：无 P0、无未修复 P1；核心正确性（规则语义等价、代价严格下降、模拟器与 cirq 一致、
动作布局、PPO 数学、端到端等价率 1.0）经独立参考实现交叉验证全部通过。**

| 严重级别 | 总数 | 已修复并验证 | 仍开放 | 说明 |
| --- | --- | --- | --- | --- |
| P0（致命：崩溃 / 语义错误 / 结果失真） | 0 | — | — | 未发现 |
| P1（重大：核心产物/指标错误） | 1 | 1（P1-1） | 0 | `best.pt` 权重与归档指标不一致，已修复 + 常驻回归 |
| P2（次要：契约/边界/文档偏差） | 5 | 4（P2-1…P2-4） | 1（P2-5） | P2-5 仅“论文用图未生成”，绘图管线实测可用 |
| P3（提示：整洁度/接口语义提醒） | 3 | 2（P3-1/P3-2 中的 tqdm、README 编号） | 1（P3-3 笔记）+ `.gitkeep` | 不影响运行与结果 |

**审计测试最终结果（冻结快照，单次全量）：`55 passed, 0 xfailed, 0 warning, 50.66 s`**
（wall 53.0 s；约束 <120 s；运行期间后台还有一次 `results/default` 正式训练占用 CPU）。

- 所有 P1/P2 回归断言均为**普通通过用例**（不再使用 xfail）：缺陷若回归会直接 FAIL；
- 审计测试运行前后对 34 个被审计文件做 SHA256 对比：**无变化**，证明结论针对同一冻结快照。

---

## 1. 审计快照（冻结，SHA256 前 16 位 + mtime）

| 文件 | mtime | SHA256（前 16 位） |
| --- | --- | --- |
| `qco/gates.py` | `2026-10-20 00:44:56` | `5f8994345e17f26f` |
| `qco/circuit.py` | `2026-10-20 00:45:13` | `ad236bffe19d2d51` |
| `qco/simulator.py` | `2026-10-20 00:44:56` | `46edd405eb31d63c` |
| `qco/observation.py` | `2026-10-20 00:45:13` | `5251c79e264f2f36` |
| `qco/generator.py` | `2026-10-20 00:45:13` | `3d66ed1402d42c90` |
| `qco/actions.py` | `2026-10-20 00:44:56` | `c577c547b2ed74d2` |
| `qco/rules.py` | `2026-10-20 00:45:31` | `f1a30950b9673312` |
| `qco/masking.py` | `2026-10-20 01:20:30` | `ad4b3ef68fa23d7f` |
| `qco/environment.py` | `2026-10-20 01:20:30` | `8d798d1a38ee11be` |
| `qco/agent.py` | `2026-10-20 00:52:21` | `7a3fb5cb808b8748` |
| `qco/ppo.py` | `2026-10-20 00:49:19` | `7f7fab4c8b9c673e` |
| `qco/rollout.py` | `2026-10-20 00:52:16` | `0fcf6d770fa61e5f` |
| `qco/baselines.py` | `2026-10-20 00:48:42` | `37b8127dacfc5982` |
| `qco/evaluate.py` | `2026-10-20 00:49:08` | `fa3487e5a66c3029` |
| `qco/trainer.py` | `2026-10-20 01:12:38` | `a1de6f9956ae3e62` |
| `qco/utils.py` | `2026-10-20 00:44:56` | `628fb061a2a69131` |
| `qco/visualizer.py` | `2026-10-20 00:49:52` | `6a4e4f016f157baf` |
| `qco/config.py` | `2026-10-20 00:58:37` | `dbbd61570d1a2e76` |
| `scripts/train.py` | `2026-10-20 01:10:56` | `4195df9c46d21205` |
| `scripts/evaluate.py` | `2026-10-20 01:03:58` | `12a133ef5b6c5371` |
| `scripts/make_dataset.py` | `2026-10-20 00:56:57` | `c84f9fc8184ede61` |
| `scripts/make_figures.py` | `2026-10-20 00:56:53` | `c32c44753d10c0db` |
| `docs/architecture.md` | `2026-10-20 01:20:47` | `433930021021401e` |
| `README.md` | `2026-10-20 01:22:42` | `16adf389464222f0` |
| `requirements.txt` | `2026-10-20 01:22:42` | `f6ae1a2d97b8c068` |
| `pyproject.toml` | `2026-10-20 01:22:42` | `cbfadd93cd0d6d09` |
| `tests/test_audit.py`（本审计交付物） | `2026-10-20 01:23:55` | `1406da6c37f6089d` |

运行环境：Python 3.14.7（项目 `.venv`）、numpy 2.5.4、torch 2.14.1+cpu、pytest 9.1.1、
cirq 1.7.0（可选依赖，已安装用于交叉验证）；无 GPU。

> 审计期间工作区由其他 owner 持续修改（01:03–01:22 间 `qco/trainer.py`、
> `qco/masking.py`、`qco/environment.py`、`scripts/train.py`、`docs/architecture.md`、
> README/requirements 等多次变化）。上表为 **experiments 冻结后** 的最终快照，
> 本报告全部数字与结论均针对该快照；最终重跑前后哈希一致。

---

## 2. 审计覆盖与证据

| 区域 | 覆盖内容 | 用例 |
| --- | --- | --- |
| **A 重写规则** | 51 个种子 × 2 变体 = **102 个独立随机电路**（`n_qubits` 2..5、`depth` 1..8、`p_two` 5%–80%）枚举 **365 个可适用动作**：电路合法、cost 严格下降、模全局相位语义等价（独立幺正矩阵 + 实现 `circuits_equivalent` 双验证）；18 组“反复随机应用合法动作直到终态”（无死循环/空动作、步数 ≤ `num_ops+depth`、终态无可行动作）；36 组掩码一致性；跨空闲层合并、乘积匹配、阻断抵消、CNOT 顺序、二比特门前后移、混排等手工边界 | `test_a1`–`test_a7` |
| **B 模拟器** | 逐门已知作用（big-endian 逐位核对）；12 个随机电路 numpy 幺正 vs **独立 kron/置换嵌入**；10 个随机电路 vs `cirq.unitary`（模全局相位，含非相邻二比特门、CNOT 反序）；全 11 门覆盖的 `to_cirq/from_cirq` 往返；8×8 单比特门乘积穷举匹配；幺正性/缓存 | `test_b1`–`test_b7` |
| **C 动作布局** | `(r,q,m)↔flat` 公式/往返/向量化；`env.valid_actions` × `unravel_index` × `decode_action` × 掩码一致；单/双合法动作下的采样、`log_prob`、熵、非法概率 0；rollout 观测/掩码/动作与逐步重放一致；done 记账 = 回合统计数；bootstrap 价值 | `test_c1`–`test_c6` |
| **D PPO 数学** | 10 组随机含 done 数据 GAE vs **独立 numpy 递推环**（horizon=1、1-D、跨环境不串扰、非法形状报错）；真实 rollout 上 ratio≡1、`log_probs` vs 独立 masked log-softmax（无双重 softmax）、概率和=1；`update` 实际使用的掩码/观测与 rollout 保存逐一相等；value loss = 独立 `MSE(V, returns)`（非 rewards）；熵指标一致；KL 早停生效；指标键/损失公式/参数更新 | `test_d1`–`test_d7` |
| **E 端到端** | 临时目录 + 子进程真实运行 `scripts/make_dataset.py`→`scripts/train.py`（3 比特/8 时隙/深度 5/2 iterations/`--dataset`）→`scripts/evaluate.py`→`scripts/make_figures.py`：6 产物齐全、CSV 列与契约一致、四方法等价率 1.0、checkpoint 可加载、evaluate 指标==summary.best_eval、5 张 PNG 生成；**P1 回归**（3 iterations+eval_every=1 重载一致性） | `test_e1`–`test_e4`、`test_g13` |
| **F 边界** | 空电路、已最优、`depth==n_moments`、`n_qubits=2`、`max_steps=1`（截断/auto-reset）、非法/越界/被掩码动作报错且状态不变、reset 幂等、训练/评估集零重叠、`data/eval.json` 可复现、电路校验/序列化/cost、三层配置校验（含 λ=0 拒绝） | `test_f1`–`test_f11` |
| **G 文档/CLI/契约** | 6 入口 `--help` 全通（含任意 CWD）；契约 §16 参数名/默认值/示例逐项核对；契约 §1–§16 公开名称存在性；dataclass 字段；README 本地链接；依赖声明与 cirq 延迟导入；契约 §18 风格（future import/无 print/docstring）；契约 §4 与实现签名一致性；契约 §8 掩码广播（含非法形状 ValueError） | `test_g1`–`test_g13` |

所有断言均与独立参考实现对照或与契约文本对照，未使用“实现输出 == 实现输出”的同义反复。

---

## 3. P1 问题（1 项，已修复并验证）

### P1-1 `best.pt` 保存“最后一轮”权重却归档“最优轮”指标

- **位置（修复前）**：`qco/trainer.py` `_periodic_evaluation` 末尾
  `self._save_checkpoint(BEST_FILENAME, self.best_iteration, self.best_eval)`，
  而 `_save_checkpoint` 使用调用当时的 `self.policy.state_dict()`（即最后一轮权重）。
- **最小复现（修复前行为）**：极小配置
  `n_qubits=3, n_moments=8, initial_depth=5, max_steps=8, iterations=3, eval_every=1`：
  `best_iteration=1`、`iterations=3`，但 `best.pt` 与 `last.pt` 的 `state_dict`
  逐键 `torch.equal == True`；`best.pt["eval"]` 记录
  `mean_final_cost=10.1667 / cost_reduction_pct=38.3838`，而重新加载 `best.pt`
  在同一评估集上评估得 `10.8333 / 34.3434`。即
  `scripts/evaluate.py --checkpoint best.pt` 报出的“最优模型”数字与
  `summary.json` 的 `best_eval` 不一致，模型选择名存实亡。
- **影响**：README/论文主结果表声称 `best.pt` 是评估集 `mean_final_cost` 最优的模型，
  实际是最终模型；最优轮选择失效。
- **修复（experiments 已冻结）**：在“刷新最优”的当下立即保存权重快照
  （`best_eval` 与权重严格对应）；`summary.results["agent"]` 与 `comparison_table`
  对齐 `best_eval`；最后一轮指标单独记录为 `summary["final_eval"]`；
  `eval_log.csv` 仍记录每轮真实评估（最后一轮 agent 行 = 最终轮真实指标）。
- **回归验证**：`test_e3_best_checkpoint_reload_matches_recorded_metrics`（常驻、非 xfail）：
  `best_iteration=1<3`（回归前置条件被显式断言），重载 `best.pt` 的
  `mean_final_cost`/`cost_reduction_pct`/`gate_reduction_pct`/`depth_reduction_pct`
  与 `summary["best_eval"]` 在 1e-6 容差内一致，`best.pt` 与 `last.pt` 权重不同；
  `test_e2` 再验证 `scripts/evaluate.py` 加载 `best.pt` 的 `evaluation.json`
  agent 指标与 `summary.best_eval` 一致；`test_e1` 验证
  `summary.results["agent"].mean_final_cost == summary.best_eval.mean_final_cost`。

---

## 4. P2 问题（5 项：4 已修复，1 开放）

### P2-1（已修复并验证）扩展实验命令缺少 `--dataset`，7 比特命令必然失败

- **修复前位置**：`qco/trainer.py` 固定读取 `PROJECT_ROOT/data/eval.json`（且按当前工作目录
  解析，换 CWD 会静默换评估集），`scripts/train.py` 没有 `--dataset`；README 的 7 比特命令
  （`make_dataset --out data/eval_7q.json` + `train.py --n-qubits 7`）会因 trainer 加载
  5 比特 `data/eval.json` 报“评估集电路比特数 5 与配置 7 不一致”。
- **附带影响**：任务书 E 要求“3 比特极小配置在临时目录跑 `scripts/train.py`”，
  修复前因 `data/eval.json` 存在会 100% 失败；审计当时只能用“等价于全新 clone”的
  子进程 patch 方式执行，并已如实上报限制。
- **修复**：`TrainConfig.eval_dataset`（相对项目根解析；显式路径必须存在，否则
  `FileNotFoundError` 给出可操作提示；`None` 表示沿用 `data/eval.json`，缺失则现场生成）；
  `scripts/train.py --dataset`；README 扩展命令同步并补充说明。
- **回归验证**：`test_e1`/`test_e2`/`test_e4` 现以**纯 CLI**
  （`scripts/train.py --dataset <tmp>/eval_tiny.json --n-qubits 3 --n-moments 8 …`）
  完成 3 比特端到端训练 + 评估，未使用任何 patch。

### P2-2（已修复并验证）`observation_shape` 签名与契约不符

- **修复前**：契约 §4 为 `observation_shape(config: EnvConfig)`，实现
  `qco/observation.py:75` 为 `observation_shape(n_qubits, n_moments)`。
- **修复**：**改契约**——`docs/architecture.md` §4 已改为
  `observation_shape(n_qubits: int, n_moments: int) -> tuple[int, int, int]`，与实现一致。
- **验证**：`test_g7_observation_shape_signature_matches_contract` 现在解析契约文本
  与 `inspect.signature` 并逐参数比对，再抽查 `observation_shape(3,5)` 与
  `encode_circuit(...).shape` 一致（常驻通过用例）。

### P2-3（已修复并验证）`masked_log_softmax` 不满足契约的“mask 可广播”

- **修复前**：`qco/masking.py` 直接 `mask.expand_as(logits)`；当 logits 为展平的
  `[B, R*Q*M]` 且 mask 为 `[R, Q, M]` 时抛 `RuntimeError`（契约 §8 允许广播）。
- **修复**：新增 `_broadcast_mask_to_logits`，支持
  `[R,Q,M] → [R*Q*M] → [B,R*Q*M]` 的行优先展平广播；`masked_log_softmax` 与
  `masked_entropy` 共用；非法形状显式抛 `ValueError`。
- **验证**：`test_g8_masked_log_softmax_accepts_documented_broadcast`（常驻通过）：
  `[R,Q,M]` 掩码 + 展平 logits 与显式展平参考逐元素一致、概率和=1、非法概率严格 0、
  `masked_entropy` 广播一致、非法形状 `ValueError`。
- **PPO 路径不受影响（已确认）**：`test_d3`/`test_d4`/`test_d5` 在真实 rollout 上验证
  采样 `log_probs` 与独立 masked log-softmax 逐项一致、更新使用的掩码/观测与 rollout
  保存完全一致、value loss/熵/ratio 数学不变；同形状（`[B,R*Q*M]`）调用行为未变。

### P2-4（已修复并验证）`lambda_depth=0` 被允许但环境崩溃

- **修复前**：`EnvConfig` 只拒绝 `lambda_depth<0`（0 合法）；移动规则只减小深度，
  λ=0 时 cost 不变，`env.step` 的内部不变量断言抛 `AssertionError`
  （契约 §7 “任一 applicable 动作 apply 后 cost 严格下降”依赖 λ>0）。
- **修复**：`EnvConfig` 要求 `lambda_depth > 0`（错误信息说明 `move_*` 规则依赖深度项）；
  契约 §9 同步注明。
- **验证**：`test_f10_lambda_zero_config_is_rejected`（常驻通过）断言 `0.0/-1e-9/-0.5`
  均抛 `ValueError` 且信息含 `lambda_depth`，默认 0.5 与极小正数合法；
  `test_f11` 记录“移动规则只改深度项、λ=0 时 cost 不变”这一依赖关系。

### P2-5（开放，文档级）文档引用的 `docs/figures/*.png` 尚未生成

- `README.md:350-364` 与 `docs/experiment.md` 引用
  `docs/figures/{training_curves,method_comparison,rule_usage,circuit_before,circuit_after}.png`，
  但 `docs/figures/` 目录当前不存在；README 已注明“若图片暂未显示，说明尚未生成后复制”。
- 审计已实际运行 `scripts/make_figures.py`（`test_g13`）确认 5 张图可正常生成
  （单次约 8–14 s），即缺失只是“尚未执行/复制”，**不是绘图代码缺陷**。
- **建议**：正式实验完成后把 `results/figures/*.png` 复制到 `docs/figures/`，
  或暂时移除图片引用。该问题不影响代码与实验结果。

---

## 5. P3 提示（3 项）

### P3-1（已修复）`tqdm` 声明但未使用
审计指出 `requirements.txt`/`pyproject.toml` 声明 `tqdm>=4.65` 但全仓库无 `import tqdm`；
docs owner 已从两处删除该依赖（现 requirements 为 numpy/torch/matplotlib，cirq 注释为可选）。

### P3-2（已修复）README 章节编号重复
审计指出 README 实验章节出现重复的 `7.4`；docs owner 已整理为 `7.1–7.5` 连续编号。

### P3-3（开放，仅供记录）契约 §8 的 4-D logits 输入路径语义提醒
`masked_log_softmax` 对 `[..., R, Q, M]` 形状的 logits 沿**最后一维（M）**做
log-softmax，而不是“整个动作空间 R×Q×M”的分类分布；内部调用方（`agent.act`、
`PPO`）都先展平为 `[B, R*Q*M]` 再调用，因此**训练/评估路径不受影响**；
仅当第三方按契约 §8 的 4-D 写法直接把它当动作分布使用时需注意。
建议后续在契约 §8 注明“4-D 输入按最后一维归一化，动作分布请传展平形状”，
或函数内部对 4-D logits 先展平。本审计 `test_g8` 已对该路径做形状/掩码级断言，
不做跨写法数值等价断言（避免把该语义写成“保证”）。

### 其他提示
- `.gitignore` 引用 `!results/.gitkeep`、`!data/.gitkeep`，但两个占位文件不存在
  （脚本 `ensure_dir`/`save_circuits` 会自建目录，功能无影响）。

---

## 6. 已验证无问题清单（重点，均为独立验证）

### A. 重写规则语义（`qco/rules.py`、`qco/masking.py`）
- 4 条规则 `name`/`rule_index`/顺序/`RULES_BY_NAME` 与契约 §0/§7 一致；
- **102 个随机电路、365 个可适用动作**逐一验证：`apply` 返回合法电路、cost 严格下降、
  与变换前模全局相位等价（独立幺正矩阵 + 实现 `circuits_equivalent` 双验证）；
- 反复随机应用合法动作必然终止：每步 cost 严格下降、步数 ≤ `num_ops+depth`、
  终态无任何可行动作（无死循环/无空动作/无“零收益”动作）；
- 掩码 `[R,Q,M]` 与逐锚点 `is_applicable` 完全一致，padding 时隙恒 False，
  `flatten/unflatten` 与 `applicable_actions`、动作解码一致；
- 手工边界：跨空闲层合并（H…Y…H）、乘积匹配（T·T→S、T·T†→I、Y·Z→X、H·S 不匹配）、
  中间门阻断抵消（同比特阻断 / 其他比特不阻断）、CNOT 比特顺序不可交换、
  二比特门前后移跨层、2 比特与单比特混排、独占层要求、深度 1 边界；
- 规则是纯函数：不原地修改输入电路、重复调用结果一致。

### B. 模拟器与 cirq（`qco/simulator.py`、`qco/circuit.py`、`qco/gates.py`）
- big-endian 正确：`X(q0)|00>=|10>`、`X(q1)|00>=|01>`、`H(q0)`、`CNOT(0,1)`、
  `CNOT(1,0)`、`SWAP`、`T/S†/CZ` 相位逐项核对；`gate_matrix_on` 与独立
  kron/置换嵌入在单/双比特、相邻/非相邻/反序位置上一致；
- 12 个随机电路的 numpy 幺正矩阵与独立参考逐元素一致（1e-10），且为幺正矩阵；
- 10 个随机电路与 `cirq.unitary` 模全局相位一致（`assert_allclose_up_to_global_phase`）；
- 全 11 种门覆盖的 `to_cirq`/`from_cirq` 往返严格相等（随机电路同样）；
  无法识别的 cirq 门显式抛 `ValueError`；
- `Circuit.unitary` 缓存生效；`match_single_qubit_gate` 对全部 8×8 单比特门乘积的
  判定（命中名字 / I / None）与门库矩阵完全一致，非幺正/错形状返回 None。

### C. 动作布局一致性（`qco/actions.py`、`qco/agent.py`、`qco/rollout.py`）
- 动作编码公式 `(r*Q+q)*M+m`、解码逆运算、向量化版本一致，负下标报错；
- `env.valid_actions` ↔ `np.unravel_index` ↔ `decode_action` ↔ 掩码索引一致，
  `env.step` 解释的 `(rule,qubit,moment)` 与之一致；
- 单合法动作观测：deterministic/采样都只能选该动作，`log_prob=0`、熵=0；
  双合法动作：概率和=1、非法概率严格 0、`log_prob`/熵与独立实现一致、采样不出界；
  全 False 掩码行显式 `ValueError`；
- rollout 逐步重放：保存观测 == `encode_circuit(replay.circuit)`、保存掩码 ==
  `replay.action_mask`、动作在保存掩码内、奖励公式复现、done 列数 == 回合统计数、
  `next_values` == 最后观测的策略价值、`EpisodeStats` 自洽
  （return=(C0-Cf)/C0、steps=规则计数和、深度/门数单调不增）。

### D. PPO 数学（`qco/ppo.py`）
- `compute_gae` 与独立 numpy 递推环在 10 组随机数据（含 done、末步 done、首步 done、
  horizon=1、1-D 输入、多环境不同 next_values）上逐项一致（1e-12）；
  `delta = r + γv(1-d) - v`、`A = δ + γλ(1-d)A'`、`returns = A + v` 均满足；
  形状不一致显式 `ValueError`；
- 真实 rollout 上：采样保存的 `log_probs` 与独立 masked log-softmax 逐项一致 ⇒
  **没有双重 softmax**；参数冻结时 `ratio ≡ 1`；概率和严格为 1；
- `PPO.update`：单小批量时捕获的掩码/观测与 rollout 保存的多重集合完全一致 ⇒
  **更新使用同一组掩码**；`approx_kl≈0`、`clip_fraction=0`、
  `policy_loss≈-mean(归一化优势)`；`value_loss` 与独立 `MSE(V, returns)`
  （**不是 rewards**）一致；熵指标与独立掩码熵一致；
  `total_loss = policy + c_v·value - c_e·entropy`；指标键 == 契约 9 项、数值有限、
  参数确实更新、KL 早停使更新步数显著小于上限。

### E. 端到端（`scripts/*`、`qco/trainer.py`、`qco/evaluate.py`）
- 3 比特极小配置（`n_moments=8, initial_depth=5, iterations=2, eval_every=1`，
  `--dataset <临时评估集>`）**纯 CLI** 跑通：`config.json`、`train_log.csv`
  （2 行、12 列与契约一致）、`eval_log.csv`（5 行；最终轮含
  noop/random/greedy/agent，全部等价率 1.0）、`best.pt`、`last.pt`、
  `summary.json` 齐全；`summary.results` 四方法 `equivalence_rate==1.0`，
  noop 零压缩、其余非负；`summary.results.agent` 与 `best_eval` 一致；
- `scripts/evaluate.py` 加载 `best.pt` 在 5 个电路上评估：退出码 0、stdout 打印
  Markdown 表、`evaluation.json` 四方法齐全且等价率 1.0，agent 指标 == `summary.best_eval`；
- P1 回归（§3）；`scripts/make_figures.py` 实际生成 5 张 PNG（>1 KB）；
- `scripts/make_dataset.py` 输出 `meta`/`circuits` 结构正确，且不修改项目 `data/`。

### F. 边界与数据
- 空电路：cost=0、掩码全 False、`valid_actions` 空、`step` 抛 `ValueError`、
  noop/random/greedy 零步不崩溃、观测全 idle；
- 已最优电路（单层两门）：掩码全 False、贪心零步、语义等价；
- `initial_depth == n_moments`、`circuit.depth == n_moments` 合法；超窗电路显式报错；
- `n_qubits=2` 时规则/掩码/策略/rollout/评估全链路可用；
- `max_steps=1`：一步后 `done=True, truncated=True, terminated=False`，
  baseline 截断标记正确，rollout 每步 done + auto-reset 正常；
- 非法/越界/负下标/被掩码动作：抛 `ValueError` 且电路与步数不变；
- `reset` 幂等（同电路两次 reset 观测/掩码/cost 完全一致、规则计数清零），
  错比特数/超深电路 reset 显式报错；
- `data/eval.json` 与文档命令（`count=100, n_qubits=5, depth=12, seed=10000`）
  生成结果**逐电路一致**；训练集（seed=0）与评估集（eval_seed=10000）零重叠、各自唯一；
- 电路构造校验（同层冲突/越界/非法门名/比特数不匹配/重复比特）、空层丢弃、
  `to_dict/from_dict` 往返、`==/hash`、ASCII 图、`cost` 公式、
  `next_op_on/prev_op_on`、`layer_qubits`；
- `EnvConfig/PPOConfig/TrainConfig` 非法值全部在构造期显式报错（含 λ≤0）。

### G. 文档 / CLI / 依赖
- 6 个入口（`scripts/*.py` 4 个 + 根目录 `train.py`/`evaluate.py` 薄封装）`--help`
  全部退出码 0，且从项目目录之外运行同样可用；
- 契约 §16 命令行参数名/默认值逐项一致（train/evaluate/make_dataset/make_figures），
  契约示例命令均可解析；
- 契约 §1–§16 公开类/函数/属性/方法全部存在；dataclass 字段齐全，
  `EnvConfig/PPOConfig/EvalResult` 与契约字段集合完全相等；
- 契约 §4 `observation_shape` 签名与实现一致（P2-2 修复后）；
- README 引用的本地文件全部存在（除 P2-5 的生成图）；
- 第三方导入全部在 requirements 声明；`cirq` 仅在函数内延迟导入（可选依赖），
  `qco` 的 gates/circuit/simulator/rules/environment/generator/actions/observation
  顶层不依赖 torch；
- 契约 §18 风格：所有 `qco/` 模块有 `from __future__ import annotations`、
  无 `print(` 调试残留、公开类/函数均有 docstring、`qco.__all__` 与真实属性一致。

---

## 7. 环境限制与说明（未跳过事项）

1. **审计期间代码被其他 owner 持续修改**：01:03–01:22 间 `qco/trainer.py`、
   `qco/masking.py`、`qco/environment.py`、`scripts/train.py`、`docs/architecture.md`、
   README/requirements 等多次变化。本报告 §1 为冻结快照；最终全量重跑前后对
   34 个被审计文件做了 SHA256 对比，**无变化**，结论针对同一快照。
2. **正式训练占用 CPU**：审计期间 `results/default` 有 200 轮正式训练在跑
   （CPU 约 90%），并在冻结后由 Lead 重启。审计**未触碰** `results/default`、`data/`；
   端到端用例全部使用临时目录 + `--dataset` 指向临时评估集；
   审计测试单次全量 50.66 s（约束 < 120 s，已满足）。
3. **任务书 E 的“n_qubits=3 极小配置跑 scripts/train.py”**：P2-1 修复前因
   `data/eval.json`（5 比特）存在会被 trainer 维度校验拒绝；修复前审计用
   “等价于全新 clone（子进程内 patch `DEFAULT_EVAL_DATASET`）”执行并如实上报限制；
   **P2-1 修复后已改为纯 CLI `--dataset <临时评估集>`**，本报告 E 项结论基于修复后路径。
4. **未执行/不适用项（非跳过，均为主动排除）**：
   - 不联网、不 `pip install`、不 `git`（按要求）；
   - 未跑 150 轮正式训练与大规模基准（长时间占用 CPU，且属 Lead/experiments 职责）；
   - 未覆盖 GPU 路径（本机 `torch.cuda.is_available() == False`）；
   - `docs/figures/*.png` 未生成（P2-5；已用 `make_figures` 实测可生成）；
   - 未运行仓库全量 `tests/`（避免与后台正式训练争抢 CPU；本审计文件自身全绿）；
   - `tests/` 在原有 15 个测试文件之外新增 `tests/test_audit.py`（本审计交付物）。

---

## 8. 审计测试使用说明

```powershell
# 全量审计（冻结快照上实测 55 passed, 0 xfailed, 0 warning, 50.66 s）
C:\Users\zjc20\quantum-circuit-optimization\.venv\Scripts\python.exe -m pytest tests/test_audit.py -q

# 只跑某一区域（示例）
... -m pytest tests/test_audit.py -q -k "a"      # A 规则
... -m pytest tests/test_audit.py -q -k "d"      # D PPO
... -m pytest tests/test_audit.py -q -k "e"      # E 端到端（含真实训练子进程）
```

- 本文件所有用例均为**普通通过断言**：`test_e3`（best.pt 一致性）、`test_e4`
  （极小配置纯 CLI）、`test_g7`（契约签名一致）、`test_g8`（掩码广播）等
  已从“已知缺陷 xfail”转为常驻回归；缺陷若回归会直接 FAIL。
- 用例编号（`test_a1`…`test_g13`）与 §2 覆盖表一一对应，便于定位。
- 审计测试只写临时目录（`tmp_path` / `TemporaryDirectory`），不污染 `results/`、
  `data/` 与项目源码。

---

## 9. 最终判定

- **未发现 P0**；**P1-1 已修复并加入常驻回归**（重载 `best.pt` 指标 == `summary.best_eval`，
  且 `best.pt` 与 `last.pt` 权重在最优轮非末轮时不同）。
- 4 个 P2 已修复并验证（`--dataset`/评估集解析、`observation_shape` 契约对齐、
  掩码广播、λ>0 校验）；仅 P2-5（论文插图未生成）为文档/交付层面的开放项，
  绘图管线已实测可用。
- 核心科学正确性经独立参考实现交叉验证：**规则保持电路语义（模全局相位）、
  每步代价严格下降、模拟器与 cirq 一致、动作布局无错位、PPO 数学正确
  （GAE/ratio/掩码/returns/无双重 softmax）、端到端等价率恒为 1.0**。
- 结论：冻结快照可以交付；建议正式实验结束后补 `docs/figures/`，并按 P3-3
  在契约中补充 4-D logits 归一化说明。

---

## 8. 审计后续处理（Lead 补充，2026-10-20）

审计结束后，Lead 已完成两项收尾，使报告中的“仍开放”项全部关闭：

- **P2-5（论文用图未生成）→ 已关闭**：正式实验完成后运行 `scripts/make_figures.py --run results/default --out results/figures`，5 张 PNG 已生成并复制到 `docs/figures/`；`tests/test_audit.py::test_g13` 覆盖的绘图管线在真实产物上验证通过。
- **P3 `.gitkeep` 占位缺失 → 已关闭**：已创建 `data/.gitkeep` 与 `results/.gitkeep`，与 `.gitignore` 的例外规则一致。
- **P3 笔记（4-D logits 归一化维度）→ 已补充契约说明**：`docs/architecture.md` §8 已注明 `masked_log_softmax` 沿最后一维归一化，采样/PPO 路径统一先展平为 `[B, R*Q*M]`。

最终交付验证（Lead 执行）：全仓 `pytest tests -q` = **237 passed in 71.44 s**；正式实验（200 iterations）产物与 `scripts/evaluate.py` 独立复评一致（agent 代价压缩 38.50%、等价率 1.0）。
