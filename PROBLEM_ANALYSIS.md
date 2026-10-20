# 原仓库问题分析（PROBLEM_ANALYSIS）

> 本文对原毕业设计仓库 `Quantum-circuit-optimization-graduation-design-main` 做逐文件审计，
> 回答三个问题：**原代码错在哪里、会造成什么后果、重写版如何解决**。
> 文中的“原仓库”指原始实现；“重写版”指当前项目 `C:\Users\zjc20\quantum-circuit-optimization`。
> 原始仓库根目录共 **9 个 Python 模块**（`agent.py`、`collect_episode_data.py`、`config.py`、
> `data_generator.py`、`environment.py`、`evaluate.py`、`optimizer.py`、`rules.py`、`train.py`），
> 外加 `problem.md` 与 `TEST/` 下的 11 个调试笔记本。

---

## 0. 结论速览

原仓库的问题**不是“训练效果不好”，而是“训练闭环根本不存在”**：

1. **优化对象错了**：`rules.py` 里的 6 条“规则”全部是往电路里**添加**固定门，不是在保持语义的前提下重写电路；
2. **状态错了**：`environment.py` 的状态张量把每个门都写进同一个（错误）时隙列，门类型还丢失大半，状态无法表示“电路当前长什么样”；
3. **奖励错了**：奖励是 `-深度 - 0.2×门数` 的绝对负值，加门只会让奖励更差，智能体的最优策略是“永远不动”；
4. **动作掩码是假的**：`ActionMask.mask()` 直接返回全 `True`，没有任何“动作是否可用”的判断；
5. **概率分布错了**：策略网络先做一次 softmax，采样时又做一次归一化，PPO 更新时再做一次 softmax/exp 归一化，同一个动作在不同阶段对应三种互相矛盾的分布，`old_log_prob` 与 `new_log_prob` 不可比，PPO 失效；
6. **GAE 漏了最后一步**，回报被打包成 batch 求和，价值网络实际上没有逐状态学习；
7. **`train.py` 切片错误 + 吞异常**：每轮更新只取到 1 条样本，`PPO.update` 因输入维度不对直接报错，而 `except Exception` 把错误打印后继续，最后照样保存一个**随机初始化**的 `rl_agent.pth`；
8. **`data_generator.py` 调用了环境根本不存在的方法**（`sample_action` / `step`），一旦使用必然 `AttributeError`；
9. **`evaluate.py` 只打印一个不可解释的负奖励总和**，没有基线、没有压缩率、没有等价性检查，且复用了上面那套坏掉的采样器；
10. **`problem.md` 指出的策略维度问题方向正确，但结论不完整**：维度顺序本身不是 bug，真正的 bug 是归一化范围与采样/对数概率分解不一致（详见 2.10）。

一句话概括：**原仓库是一个“能 import、能打印 Epoch complete”的演示脚本，不是一个可训练、可评估、可复现的强化学习系统。**

| 严重级别 | 含义 | 涉及文件 |
| --- | --- | --- |
| **P0（阻断）** | 导致训练/优化在语义上不成立，或产生看似成功实则无效的结果 | `rules.py`、`environment.py`、`optimizer.py`、`collect_episode_data.py`、`train.py`、`agent.py`、`data_generator.py` |
| **P1（正确性）** | 不直接崩溃，但指标失真、不可复现、结果不可信 | `evaluate.py`、`config.py`、`collect_episode_data.py`、`agent.py` |
| **P2（工程性）** | 可维护性、依赖声明、文档、测试、异常处理 | 全部文件、`TEST/*.ipynb`、`__pycache__/` |

---

## 1. 审计范围与方法

- **审计对象**：原仓库 9 个 `.py`、`problem.md`、`TEST/` 下 11 个 `.ipynb`；
- **数量说明**：任务描述中写作“10 个 .py”，但原仓库根目录实际只有 9 个 `.py`（没有 `__init__.py`，
  `TEST/` 下也只有 notebook、没有 `.py`）。本分析以实际目录为准，逐文件覆盖全部 9 个模块；
- **审计方式**：逐行静态阅读 + 形状/接口推演；对 PyTorch 的关键行为（张量广播赋值、`Conv2d` 输入必须是 4 维、`t[i::n]` 切片语义）用当前虚拟环境做了最小验证，确认问题不是“看起来像”而是真实存在；
- **未修改原仓库**：本文件只做分析，所有修复都在重写版中完成；
- **环境**：Windows 11 + Python 3.14.7 + numpy 2.5.4 + torch 2.14.1（CPU）+ cirq 1.7.0。

补充一个关键事实：**原环境从空电路开始**。`QuantumCircuitSimulator.reset()` 把 `cirq.Circuit()` 置空，`rules.py` 的规则再一步步加门，所以原系统从未在任何“待优化的电路”上做过优化——它只是在空电路上随机堆门。

---

## 2. 逐文件分析

### 2.1 `agent.py` —— 策略头输出的不是“联合动作分布”

**文件规模**：67 行；核心类 `CircuitOptimizerAgent(nn.Module)`。

#### 问题 1（P0）：网络内部先 softmax，破坏“策略只输出 logits”的约定

```python
# agent.py:60-63
policy = self.policy_linear(x_flat)
policy = policy.view(-1, self.n_rules, self.n_qubits * self.n_moments)  # 修正此处的维度
policy = F.softmax(policy, dim=-1)
```

这里是**双重归一化的起点**：网络对“每个规则内部的 `Q×M` 个位置”各做一次 softmax，得到 `n_rules` 个和为 1 的分布，而不是一个在 `R×Q×M` 联合动作空间上和为 1 的分布。

- **后果**：上层 `collect_episode_data.py` 把 `[R, Q*M]` 展平后交给 `Categorical` 采样，`Categorical` 会**再归一化一次**；PPO 更新时 `optimizer.py` 又会对这个“概率”做 `exp` + 归一化（见 2.4）。同一个动作的 `old_log_prob` 和 `new_log_prob` 来自三种不同的分布，PPO 的重要性采样比率 `ratio = exp(new - old)` 没有意义，裁剪目标失效。
- **重写版如何解决**：网络只输出 **logits** `[B, R, Q, M]`，归一化只发生在 `qco/masking.py::masked_log_softmax` 这一处，且只对**展平后的有效动作集合**做一次；`qco/agent.py::act` 负责“掩码 → log_softmax → 采样/argmax → 返回 log_prob、value、entropy”。测试 `tests/test_agent.py` 断言非法动作概率严格为 0、`deterministic=True` 时选择最大概率动作、梯度可反传。

#### 问题 2（P0）：没有动作掩码接口

策略网络完全不接收掩码，也没有 `act()` 之类的方法；掩码是在环境采样脚本里“事后”乘上去的（见 2.5），且那个掩码本身恒真。这使“策略分布”与“合法动作集合”两个概念从一开始就没有对齐。

- **重写版**：`CircuitPolicyNet.forward()` 输出 logits，`act(observations, masks)` 显式接收掩码；`PPO.update()` 在重算 `log_prob` 时复用同一组 mask（来自 `RolloutBatch.masks`），保证策略损失与采样分布一致。

#### 问题 3（P1）：BatchNorm 定义了却从未使用；若启用也不适合 PPO

```python
# agent.py:24-28
self.conv1 = nn.Conv2d(n_gate_classes, 64, kernel_size=3, padding=1)
...
self.bn = nn.BatchNorm2d(256)  # 将批归一化层移到这里初始化
```

`self.bn` 在 `forward()` 中被注释掉（`#x = self.bn(x)`，第 54 行），成为**永远不会更新的死参数**；若按注释启用，PPO 采集/小批量更新阶段分布持续变化，BatchNorm 的滑动统计量会让价值/策略估计不稳定。

- **重写版**：主干用 `Conv2d + ELU`（三组），策略头 1×1 卷积，价值头 `Conv→ELU→Flatten→Linear→ELU→Linear`，**禁止 BatchNorm**；这一点写入了 `docs/architecture.md` 与网络 docstring，`tests/test_agent.py` 检查前向形状与梯度。

#### 问题 4（P1）：价值头输出未压缩、接口不明确

```python
# agent.py:38-39
self.value_linear = nn.Linear(flattened_size, 1)
...
# agent.py:64-66
value = self.value_linear(x_flat)
return policy, value      # value: [B, 1]
```

`value` 返回 `[B,1]`，而采集脚本把它塞进 `torch.empty(max_steps)` 的一维张量里（靠广播“碰巧”能写进去），后续计算回报时形状语义混乱。

- **重写版**：`value` 严格为 `[B]`；`act()`/`forward()` 的返回形状写入契约并由测试断言。

#### 问题 5（P1）：硬编码 batch=1，无法并行环境

`view(-1, self.n_rules, Q*M)` 中的 `-1` 与采集脚本里的 `policy.view(N_RULES, Q*M)` 相互假设 batch=1；一旦想并行多环境采样就全部失效。

- **重写版**：所有张量保留 batch 维 `[B, ...]`；`RolloutBatch` 用 `[num_envs, horizon, ...]` 存储，PPO 展平后小批量更新。

#### 问题 6（P2）：没有设备/种子/配置

网络不处理 `device`、不设种子、没有 checkpoint 配置。

- **重写版**：`qco/utils.py::set_seed/resolve_device`、`qco/trainer.py::TrainConfig`、`scripts/train.py --seed --device` 统一管理。

---

### 2.2 `environment.py` —— 状态、奖励、终止、掩码全部失真

**文件规模**：118 行；包含 `QuantumCircuitSimulator`、`QuantumCircuitEnvironment`、`ActionMask`。

#### 问题 1（P0）：状态张量的时隙索引是错的

```python
# environment.py:33-40
for op in self.circuit.all_operations():
    qubits = [q.row for q in op.qubits]
    gate_type = self.get_gate_type(op.gate)
    if gate_type is not None:
        for qubit in qubits:
            # 使用时间索引和门索引进行填充
            moment_index = min(self.n_moments - 1, len(self.circuit))
            state_tensor[qubit, moment_index, gate_type] = 1.0
```

`len(self.circuit)` 是**电路当前的时刻总数**，不是这个门所在的时隙下标。更糟的是它被 `min(n_moments-1, ·)` 截断：电路深度为 10 时，所有门都被写进第 10 列（而合法下标最大是 9）；深度不满 15 之前，所有门又都写进 `len(circuit)` 这一列；随着电路增长整列还会“漂移”。**最终状态张量只保留最近一次统计的某一列，时间信息完全丢失。**

- **后果**：状态不是马尔可夫充分的，策略无法判断“哪个门在哪个时隙、哪两个门相邻”，调度类（移动门）动作无从谈起；网络看到的状态与真实电路脱节。
- **重写版**：`qco/observation.py::encode_circuit` 逐层枚举 `moment`，显式写入 `[N_CHANNELS, n_qubits, n_moments]`；空闲比特写入独立的 `IDLE_CHANNEL`，超出深度的列全部标记为空闲，保证观测形状固定；`tests/test_observation.py` 逐通道断言编码正确。

#### 问题 2（P0）：门类型编码丢门

```python
# environment.py:44-55
def get_gate_type(self, gate):
    gate_types = ['RZ', 'PX', 'CNOT', 'SWAP']
    if isinstance(gate, cirq.ops.ZPowGate):
        return gate_types.index('RZ')
    ...
    return None
```

原实现只认 4 类门：`H`（`HPowGate`）、`S/T` 等虽然会被 `ZPowGate` 分支吞进 `RZ`，但 `H`、`Y`、`CZ`、`Sdag` 等无法识别而返回 `None`，随后被第 36-40 行**静默跳过**。此外 `n_gate_classes=4` 与重写版 11 种门 + 3 个辅助通道（控制/目标/空闲）完全不是一个量级。

- **后果**：状态张量丢失信息；网络输入通道与物理门集不对应。
- **重写版**：门库固定为 `X,Y,Z,H,S,Sdag,T,Tdag,CNOT,CZ,SWAP`（11 个），观测通道 14 个（11 门 + control + target + idle），顺序写入 `qco/gates.py` 且由测试锁定，禁止调换。

#### 问题 3（P0）：动作里的“位置”被直接丢弃

```python
# environment.py:97-105
def apply_rule(self, action):
    """Apply circuit transformation rule."""
    rule_index, _ = action  # 解包元组
    ...
    rule = self.rules[rule_index]
    state, reward, done = self.simulator.apply_rule(rule)
```

动作是 `(rule_index, qubit_moment_index)` 二元组，但环境只取 `rule_index`，第二个分量用 `_` 丢掉。也就是说“在哪个比特、哪个时隙执行规则”对结果没有任何影响；而 `rules.py` 的规则又都是全局加门（见 2.3），最终动作空间退化成“选 6 个全局函数之一”。

- **重写版**：动作是 `(规则 r, 量子比特 q, 时隙 m)` 三元组，统一编码为单一整数 `index = (r*Q + q)*M + m`（`qco/actions.py`），环境 `step(action)` 先解码、再查掩码、再调用 `rule.apply(circuit, qubit, moment)`；非法动作抛 `ValueError`。

#### 问题 4（P0）：奖励是绝对负代价，无法学习

```python
# environment.py:64-69
def compute_reward(self):
    depth = len(self.circuit)
    gate_count = sum(len(op.qubits) for op in self.circuit.all_operations())
    reward = -depth - 0.2 * gate_count
    return reward
```

这是“当前电路有多差”的**绝对负值**，不是“这一步变好了多少”的增量。由于所有规则都是加门，**每执行一步奖励都会更负**；在折扣回报目标下，最优策略是“什么都不做”。同时该奖励的量级随电路规模变化，不同回合之间不可比，也没有用初始代价归一化。

- **重写版**：采用**势函数塑形**：`r_t = (C(s_t) - C(s_{t+1})) / C(s_0)`，`C = 门数 + λ·深度`。每步严格为正；回合总回报恰好等于 `(C(s_0) - C(s_T))/C(s_0)`（望远镜求和），即“相对初始代价的压缩比例”，与回合长度、绝对规模无关。`qco/environment.py::step` 与 `tests/test_environment.py` 验证该公式。

#### 问题 5（P0）：没有终止条件

```python
# environment.py:71-74
def check_done(self):
    """Check if optimization is complete."""
    # Implement custom termination criteria...
    return False
```

`done` 永远为 `False`。回合只能靠外部 `max_steps` 截断；采集脚本遇到截断时又没有 bootstrap，GAE 的价值估计有偏。

- **重写版**：`done = (无任何可用动作) or (steps >= max_steps)`；`info` 里区分 `terminated`（真正无动作可做）与 `truncated`（达到步数上限）；`RolloutCollector` 为每步保存 `next_values` 做 bootstrap，`PPO.compute_gae` 用 `(1-done)` 截断，auto-reset 不影响优势估计。

#### 问题 6（P1）：`ActionMask` 恒为全 1

```python
# environment.py:114-118
def mask(self, circuit, gate_classes):
    """Compute the action mask based on the current circuit state."""
    mask = np.ones((self.n_rules, self.n_qubits * self.n_moments), dtype=bool)
    # Implement masking logic based on valid circuit transformation rules
    return torch.tensor(mask, dtype=torch.float32)
```

注释直接写着 “Implement masking logic...” —— **从未实现**。全 1 掩码意味着策略会在大量无意义动作上分配概率，`select_action` 里的“全被屏蔽则随机选”分支也永远不会触发（死代码）。

- **重写版**：`qco/masking.py::build_action_mask` 遍历 `RULES[r].is_applicable(circuit, q, m)`，返回 `[R, Q, M]` 布尔掩码；`masked_log_softmax` 让非法动作概率严格为 0；`masked_entropy` 只对有效动作求熵；`tests/test_masking.py` 与 `applicable_actions()` 交叉验证。

#### 问题 7（P1）：状态/接口形状约定不一致

`get_state()` 返回 `[1, C, Q, M]`（带 batch 维），`reset()` 直接返回它；`apply_rule` 返回三元组 `(state, reward, done)`；而采集脚本预分配的是**不带 batch 维**的 `[T, C, Q, M]`，靠 PyTorch 的 `copy_` 广播“碰巧”能赋值（本次审计已用最小实验确认 `states[0] = state` 在 `[1,4,5,15] → [4,5,15]` 时确实能广播成功），从而**掩盖了接口不一致**。一旦环境改成并行或带真实 batch 就会炸。

- **重写版**：统一的 Gym 风格接口 `reset() -> obs`、`step(action) -> (obs, reward, done, info)`；观测固定 `[C, Q, M] float32`，采样器负责加 batch 维；所有形状写入 `docs/architecture.md` 契约。

#### 问题 8（P2）：从空电路开始，没有随机初始电路/种子

`reset()` 只是 `cirq.Circuit()`；没有随机初始电路生成逻辑，也没有 RNG/seed，实验不可复现。

- **重写版**：`EnvConfig.initial_depth=12`、`p_two=0.15`，`reset()` 用 `qco/generator.random_circuit` 生成深度 12 的随机电路；`seed` 固定；数据集生成与训练/评估使用不同种子，避免泄漏。

---

### 2.3 `rules.py` —— “规则”是在加门，不是在优化

**文件规模**：63 行；6 个规则函数 + `RULES` 列表。

#### 问题 1（P0）：4 条“规则”都是往固定比特加固定门

```python
# rules.py:6-11
def rz_rule(simulator):
    """Apply RZ gate to the first qubit."""
    qubits = simulator.qubits
    gate = cirq.rz(np.pi / 2)
    simulator.add_gate(gate, (qubits[0],))
    return simulator
```

`rz_rule` / `x_rule` / `cnot_rule` / `swap_rule` 分别给 q0 加 RZ、给 q0 加 X、给 q0–q1 加 CNOT、加 SWAP。这些操作**改变电路语义**（幺正矩阵变了），既没有“目标电路”，也没有“等价”概念，更不会降低任何代价——门越加越多。

- **后果**：强化学习优化的目标函数不存在；即使训练成功，也只是学会了如何往空电路上堆门。
- **重写版**：4 条规则全部是**语义保持的等价重写**：`MergeSingleQubitRule`（合并相邻单比特门、单位阵删除）、`CancelTwoQubitRule`（相同自逆二比特门抵消）、`MoveEarlierRule` / `MoveLaterRule`（独占时隙的门前后移动压缩深度）。每条规则都满足“应用后 `cost` 严格下降 + 幺正等价（模全局相位）”，由 `tests/test_rules.py` 对 200 个随机电路 × 4 条规则逐一验证。

#### 问题 2（P0）：`commute_rule` 盲目交换前两个门

```python
# rules.py:34-40
def commute_rule(simulator):
    """Swap the order of two adjacent gates if commutable."""
    qubits = simulator.qubits
    if len(simulator.circuit) >= 2:
        op1, op2 = list(simulator.circuit.all_operations())[:2]
        simulator.circuit = cirq.Circuit([op2, op1] + list(simulator.circuit.all_operations())[2:])
    return simulator
```

名字叫 “if commutable”，但代码**没有任何对易判断**：直接把前两个操作交换。若 `op1=CNOT(q0,q1)`、`op2=X(q0)`，交换后电路语义完全改变。另外用 `cirq.Circuit([...])` 重建电路会按 cirq 的默认策略重新排布时刻，连原来的时间结构也被破坏。

- **重写版**：不存在“任意交换”规则。移动类规则只在 `op.qubits` 之外的层之间移动（这些层上的门与该门作用比特不相交，天然对易），且保持每个比特上的门顺序不变；`MoveEarlierRule` / `MoveLaterRule` 的 `is_applicable` 强制 `len(layer) == 1` 以保证深度下降。

#### 问题 3（P0）：`cancel_adjacent_rz` 不检查是否同一比特

```python
# rules.py:42-61
def cancel_adjacent_rz(simulator):
    """Cancel adjacent RZ gates if their angles sum to zero."""
    new_ops = []
    for op in simulator.circuit.all_operations():
        if not new_ops or not isinstance(op.gate, cirq.ZPowGate):
            new_ops.append(op)
        else:
            prev_op = new_ops.pop()
            if isinstance(prev_op.gate, cirq.ZPowGate):
                angle_sum = prev_op.gate._exponent + op.gate._exponent
                if np.isclose(angle_sum % 2, 0):
                    continue  # Cancel out
                ...
```

这里把 `all_operations()` 展平后比较“相邻两个操作”，**完全没有检查它们是否作用在同一个量子比特上**：`RZ(a)@q0` 后面紧跟 `RZ(b)@q1` 也会被当成可抵消对；抵消条件只看指数和是否模 2 为 0。另外直接访问 cirq 私有属性 `_exponent`，跨版本不可靠。

- **重写版**：`CancelTwoQubitRule` 只处理 `CNOT/CZ/SWAP` 三种自逆二比特门；要求两个门是**同一层、同一个门**，且两比特之间没有任何中间门（用 `next_op_on` 分别检查两个比特）；删除后语义严格不变。单比特门的任意角度合并不在本设计范围内（见 2.10 与 `docs/design.md` 的“局限”）。

#### 问题 4（P0）：没有“适用性”与“动作位置”，`RULES` 只是 6 个全局函数

`RULES` 是 `[rz_rule, x_rule, cnot_rule, swap_rule, commute_rule, cancel_adjacent_rz]`，每个函数只接收 simulator；没有 `(qubit, moment)` 参数，没有 `is_applicable`，也没有“该规则在这个位置是否有定义”的概念。`ActionMask` 因此无从实现（见 2.2 问题 6）。

- **重写版**：抽象基类 `RewriteRule` 定义 `name/description/rule_index/is_applicable(circuit, qubit, moment)/apply(...)`；`RULES` 顺序即动作空间规则下标（0–3，禁止调换）；`applicable_actions(circuit)` 枚举所有可行动作；掩码、环境、网络、测试全部基于同一份规则定义。

#### 问题 5（P2）：规则层绑定 `cirq`

规则直接构造 `cirq.rz/cirq.X/...`，没有 cirq 就无法运行；而原仓库又没有声明依赖文件。

- **重写版**：门库用 numpy 矩阵定义（`qco/gates.py`），核心环境/规则/模拟器零 cirq 依赖；cirq 仅作为可选互操作（`to_cirq/from_cirq`、测试交叉验证）。

---

### 2.4 `optimizer.py` —— 名为 PPO，实际几乎每个环节都错

**文件规模**：151 行；类 `PPO`。

#### 问题 1（P0）：GAE 漏掉最后一步

```python
# optimizer.py:97-106
advantages = torch.zeros_like(rewards)
gae_lambda = 0.95
last_gae_lam = 0

# 修改循环，使其不包括最后一个元素，避免索引越界
for t in reversed(range(len(rewards) - 1)):  # 减1以避免索引越界
    next_non_terminal = 1.0 - dones[t+1]
    delta = rewards[t] + self.gamma * values[t+1] * next_non_terminal - values[t]
    advantages[t] = delta + self.gamma * gae_lambda * next_non_terminal * last_gae_lam
    last_gae_lam = advantages[t]
```

循环故意排除了最后一个时间步，注释把“避免越界”当成了正确性。结果是**最后一步优势恒为 0、回报恒等于价值**；长度 1 的 batch 甚至一次循环都不进，优势全 0。同时没有 bootstrap 价值，截断/终止混杂，价值估计有偏。

- **重写版**：`PPO.compute_gae(rewards, values, dones, next_values, gamma, gae_lambda)` 反向递推**包含最后一步**，`v_{t+1}` 在 `t` 为最后一步时取 `next_values[env]`：
  - `delta_t = r_t + γ·v_{t+1}·(1-d_t) - v_t`
  - `A_t = delta_t + γλ·(1-d_t)·A_{t+1}`
  - `returns = advantages + values`
  `tests/test_ppo.py` 用手算/逐项参考循环校验。

#### 问题 2（P0）：同一个动作被归一化三次

`update()` 里拿到的 `new_policy` 已经是 `agent.py` softmax 过的**概率**；`compute_log_probs()` 又对概率做 `exp` 再逐规则归一化：

```python
# optimizer.py:134-148
rule_prob = policy[i, rule_indices[i]]
unnormalized_probs = torch.exp(rule_prob)      # 对概率取 exp，不是 logits
probs = unnormalized_probs / unnormalized_probs.sum(dim=0, keepdim=True)
...
dist = Categorical(probs=probs)
```

并且在 `update()` 第 72 行还有一次 `F.softmax(new_policy, dim=-1)`。**同一个策略在同一批数据上被解释成三个不同的分布**：

1. 采样时的分布：softmax(logits) 逐规则归一化后再乘掩码、再由 `Categorical` 对 `R×Q*M` 联合归一化；
2. `old_log_prob`：上面那个联合分布的 log 概率；
3. `new_log_prob`：对“概率”再 `exp`、再逐规则归一化后的 `Categorical(probs)` 的 log 概率。

三者不是同一个分布，`ratio` 即失去意义，PPO 的裁剪目标不再是对同一策略的近似约束，梯度方向错误。更隐蔽的是：**概率为 0 的非法动作经过 `exp(0)=1` 后反而获得很大权重**——掩码在数学上被“反向”破坏了。

- **重写版**：网络输出 logits；`masked_log_softmax` 只做一次，采样与 `log_prob` 共用同一结果；`PPO.update` 用 `ratio = exp(new_log_probs - old_log_probs)`，其中二者都是“同一掩码下的联合动作分布”的 log 概率；不做任何二次 softmax/exp 归一化。这是 `docs/architecture.md` 的硬性要求，也是 `tests/test_ppo.py` 的审计点（`ratio` 初值应为 1）。

#### 问题 3（P0）：采样与对数概率的分解不一致

采样端把 `[R, Q*M]` 展平后在**联合空间**上采样；`compute_log_probs` 端却把动作拆成 `rule_index` 与 `qubit_moment_index` 两段，按“先规则、再规则内位置”的方式计算概率（`optimizer.py:124-148`）。两段概率之积并不是采样分布的概率，除非两段恰好条件独立，而这里不是。

- **重写版**：动作统一编码为单一整数 `index`（`qco/actions.py`），采样与 `log_prob` 都直接取展平联合分布的第 `index` 项，不存在两段式分解歧义。

#### 问题 4（P0）：回报与价值被按 batch 求和，价值学习失效

```python
# optimizer.py:60-68
returns = returns.sum(dim=0)                 # 一维时直接压成标量
returns = returns.clone().detach().unsqueeze(0)
new_values_aggregated = new_values.sum(dim=0)
value_loss = F.mse_loss(new_values_aggregated, returns)
```

`returns` 与 `new_values` 被沿着 batch 维求和，价值网络被要求拟合“整个 batch 的回报总和”，而不是逐状态拟合各自回报。批大小、回合长度一变，目标就变；优势与价值完全脱钩。

- **重写版**：`returns = advantages + values` 与原张量同形（逐环境、逐时间步）；展平后对每条样本做 MSE；`explained_variance` 作为监控指标输出。

#### 问题 5（P0）：不是 PPO —— 没有 epoch、minibatch、KL 早停、梯度裁剪

```python
# optimizer.py:79-82
self.optimizer.zero_grad()
total_loss.backward()
self.optimizer.step()
```

每次 `update()` 只做一次梯度更新，没有多轮 epoch、没有小批量打乱、没有 `target_kl` 早停、没有 `max_grad_norm` 裁剪、没有学习率调度。PPO 相对于 A2C 的核心优势（同一批数据多轮利用 + 信赖域约束）完全不存在，退化成“单步策略梯度”。

- **重写版**：`PPOConfig`（`lr/gamma/gae_lambda/clip_epsilon/value_coef/entropy_coef/max_grad_norm/update_epochs/minibatch_size/target_kl/normalize_advantage`）；`update()` 内多轮 minibatch，KL 超阈值提前停止，输出完整指标字典；`tests/test_ppo.py` 检查指标键、参数确实更新、KL 早停不报错。

#### 问题 6（P1）：熵的计算对象错误

```python
# optimizer.py:72-74
new_policy_normalized = F.softmax(new_policy, dim=-1)   # 对概率再做 softmax
entropy = Categorical(probs=new_policy_normalized).entropy().mean()
```

熵是在“二次 softmax 后的位置分布”上算的，既不是原始策略的熵，也没有覆盖规则维；`entropy_coef` 的探索鼓励作用失真。

- **重写版**：`masked_entropy(log_probs, mask)` 直接由带掩码的 `log_probs` 计算 `-Σ p·log p`，非法动作贡献为 0；在 `PPO.update` 中对有效动作计算平均熵。

#### 问题 7（P1）：`advantages` 归一化在错误维度上

```python
# optimizer.py:50-52
mean_advantages = advantages.mean(dim=0 if advantages.dim() == 2 else None)
std_advantages = advantages.std(dim=0 if advantages.dim() == 2 else None, unbiased=False)
```

对一维张量 `dim=None` 会在整个 batch 上求均值/标准差，当 batch 只含 1 条样本时标准差为 0，优势被除以 `1e-8` 放大成噪声；同时没有处理“全零优势”的退化情形。

- **重写版**：优势在展平后的样本集合上归一化，且仅在样本数 > 1 时执行；`normalize_advantage` 可配置；`tests/test_ppo.py` 覆盖单样本退化。

#### 问题 8（P1）：超参数硬编码/来源混乱

`gae_lambda=0.95` 写死在函数里；`gamma/clip_epsilon/lr` 从 `config.py` 全局导入；`value_coef=0.5`、`entropy_coef=0.01` 写死在 `total_loss` 里；没有 `PPOConfig`。

- **重写版**：全部收敛到 `PPOConfig` dataclass，并可由 `TrainConfig`/CLI 覆盖。

---

### 2.5 `collect_episode_data.py` —— 采集的数据与策略分布对不上

**文件规模**：159 行；`select_action` 与 `collect_episode_data`。

#### 问题 1（P0）：保存的是 `s_{t+1}`，不是 `s_t`

```python
# collect_episode_data.py:130-140
next_state, reward, done = env.apply_rule(action)
state = next_state

# 填充收集到的数据
states[step_count] = state
actions[step_count] = torch.tensor(action)
rewards[step_count] = reward
dones[step_count] = done
old_log_probs[step_count] = old_log_prob
values[step_count] = value
```

先执行动作、把 `state` 换成 `next_state`，再把新状态存进 `states[step_count]`；而 `value` 是对**动作前**状态 `s_t` 的估计。于是这批数据里 `(state, action, value)` 三者错位，策略梯度用 `s_{t+1}` 去评估 `a_t`。

- **重写版**：`RolloutCollector.collect()` 严格按 `obs_t → policy → action → env.step → obs_{t+1}` 记录；`observations[t]`、`actions[t]`、`values[t]` 与产生该动作时的状态一致；mask 也按同一时刻保存。

#### 问题 2（P0）：没有 bootstrap / 截断处理

`while step_count < max_steps` 循环里遇 `done` 直接 `break`，既不区分终止与截断，也不为最后一步计算 `next_value`；而环境 `done` 又恒为 `False`（2.2 问题 5），所以永远是“截断但不 bootstrap”。

- **重写版**：`RolloutBatch.next_values[env]` 保存最后一步之后观测（新回合初始观测）的 bootstrap 价值；`dones` 明确标记终止/截断；GAE 用 `(1-done)` 截断，auto-reset 不污染优势。

#### 问题 3（P0）：掩码采样路径不可靠，fallback 的 `old_log_prob` 是错的

```python
# collect_episode_data.py:44-56
try:
    masked_policy = masked_policy.view(-1)
    if masked_policy.sum() == 0:
        allowed_indices = torch.nonzero(action_mask.mask(env.simulator.circuit, n_gate_classes)).flatten()
        action_index = _choose_random_action(allowed_indices)
        old_log_prob = torch.tensor(0.0)     # 这不是对数概率
    else:
        action_dist = Categorical(masked_policy)
        ...
```

掩码恒为全 1（2.2 问题 6），所以 `sum()==0` 分支是死代码；一旦真的走到该分支，`old_log_prob = 0` 意味着“该动作概率为 1”，与随机采样的事实矛盾，会直接污染 PPO 的 `ratio`。另外 `masked_policy` 是“逐规则 softmax 后又乘掩码”的概率，`Categorical` 内部还会再归一化一次（第三次归一化，见 2.4 问题 2）。

- **重写版**：采样在 `CircuitPolicyNet.act(obs, mask)` 内部完成，`log_prob` 与采样来自同一 `masked_log_softmax`；环境在无可用动作时直接给出终止，不存在“全部被掩码还要硬选”的状态。

#### 问题 4（P1）：batch=1 与形状假设贯穿全文件

`policy.view(N_RULES, N_QUBITS * N_MOMENTS)`（第 111 行）假设单样本；`state_shape = env.simulator.get_state().shape[1:]`（第 92 行）把 `[1,C,Q,M]` 去掉 batch 维后预分配 `[T,C,Q,M]`，再靠广播把 `[1,C,Q,M]` 写进 `[C,Q,M]`（本次审计已验证这种广播会成功），掩盖了接口错误。没有并行环境、没有掩码存储、没有 `next_values`。

- **重写版**：`RolloutBatch` 显式定义 `observations [num_envs, horizon, C, Q, M]`、`actions/log_probs/values/rewards/dones [num_envs, horizon]`、`masks [num_envs, horizon, R, Q, M]`、`next_values [num_envs]`；并行环境按 horizon 采样，回合结束 auto-reset。

#### 问题 5（P2）：宽泛的 try/except 包裹后打印再抛出

外层 `except Exception as e: print(...); raise` 只增加了噪音，没有提供任何恢复或上下文；与 `train.py` 的“吞异常”形成对比（那边是吞掉继续）。

---

### 2.6 `train.py` —— 切片错误 + 吞异常 = 从未训练

**文件规模**：66 行；脚本式顶层代码。

#### 问题 1（P0）：minibatch 切片写错，每次只拿到 1 条样本

```python
# train.py:38-47
# 将数据按步骤划分并批量更新模型
for step in range(STEPS_PER_EPOCH):
    batch_states = states[step::STEPS_PER_EPOCH]
    batch_actions = actions[step::STEPS_PER_EPOCH]
    ...
    ppo.update(batch_states, batch_actions, batch_rewards, batch_dones, batch_old_log_probs, batch_values)
```

`states` 形状是 `[T, C, Q, M]`（T=10），`states[step::10]` 的语义是“从 step 开始、每隔 10 个取一个”，结果每个 batch 只有 **1 条时间步**，且丢失了时间维/批次维。正确做法是“把 `T` 条样本按 `T×num_envs` 展平后切成大小为 `minibatch_size` 的块”，或至少 `states[step*B:(step+1)*B]`。

- **后果**：PPO 每次只看到一条样本 → GAE 无递推、优势为 0、价值损失近似常数；如果某回合提前结束（`T<10`）还会切出空张量，`mean()` 得到 `nan`。
- **重写版**：`Trainer` 按 `num_envs × steps_per_iter` 收集固定大小的 rollout，展平后按 `minibatch_size` 打乱切块，多轮 epoch 更新；`tests/test_trainer.py` 用极小配置跑通 2 轮并检查产物。

#### 问题 2（P0）：`except Exception` 吞掉一切错误，继续“训练”

```python
# train.py:50-62
    except Exception as e:
        error_message = f"""
        An error occurred during training:
        ...
        """
        print(error_message)

    print(f"Epoch {epoch + 1}/{EPOCHS} complete.")
```

上面所有错误（切片导致的 3 维输入、空 batch、形状不匹配）都被捕获、打印，然后循环继续；训练结束照常保存模型：

```python
# train.py:64-66
torch.save(agent.state_dict(), 'rl_agent.pth')
print("Training Complete. Model saved to 'rl_agent.pth'")
```

**最终产物是一个随机初始化、从未被有效梯度更新过的模型，但终端会打印 “Training Complete”。**

- **重写版**：训练错误直接抛出（fail loud），由测试和 `--help` 暴露；`Trainer.train()` 返回 summary 并写 `train_log.csv/eval_log.csv/best.pt/last.pt/summary.json`；`tests/test_trainer.py` 断言产物齐全。

#### 问题 3（P1）：没有评估、日志、checkpoint、种子、设备、CLI

只有 `trange(EPOCHS)` 和零散 `print`；没有验证集、没有 best checkpoint、没有 CSV/图表、没有 `argparse`、没有 `if __name__ == "__main__"`（import 即执行）。

- **重写版**：`scripts/train.py` argparse；`qco/trainer.py` 周期评估、best/last checkpoint、CSV 日志、summary.json；`qco/utils.set_seed`；根目录 `train.py` 为薄封装。

---

### 2.7 `data_generator.py` —— 调用了不存在的 API

**文件规模**：25 行；类 `DataGenerator`。

#### 问题 1（P0）：`sample_action` / `step` 在环境里不存在

```python
# data_generator.py:14-18
for i in range(batch_size):
    state = self.env.reset()
    action = self.env.sample_action()          # AttributeError
    next_state, reward, done = self.env.step(action)   # 同样不存在
```

原环境只有 `reset()` 和 `apply_rule(action)`，没有 `sample_action` 与 `step`。第一次 `generate_data()` 就会 `AttributeError`。（这个类从未被 `train.py`/`evaluate.py` 导入，属于“写了但完全没接上”的死代码。）

- **重写版**：数据集由 `qco/generator.py::random_circuits` + `save_circuits/load_circuits` 负责；`scripts/make_dataset.py` 生成 `data/eval.json`；训练器按种子生成训练集、从文件加载评估集。

#### 问题 2（P1）：输入形状与状态形状不匹配

```python
# data_generator.py:11
inputs = np.zeros((batch_size, self.env.n_qubits, self.env.n_gate_classes, self.env.n_moments))
```

期望 `[B, Q, C, M]`，而环境状态是 `[1, C, Q, M]`（且带 batch 维）；赋值 `inputs[i,:,:,:] = state.numpy()` 形状对不上。即使能跑，通道顺序也与网络输入不一致。

- **重写版**：数据集是“电路 JSON”（`meta + circuits`），观测张量在 `env.observation` 中按统一 `[C,Q,M]` 编码生成，不重复造形状。

#### 问题 3（P2）：`targets=[reward, done]` 不是可学习的监督目标

没有动作标签、没有价值目标；用预测 `(reward, done)` 训练的模型与 PPO 无关。

---

### 2.8 `evaluate.py` —— 输出一个不可解释的负奖励总和

**文件规模**：33 行；脚本式顶层代码。

#### 问题 1（P0）：复用坏掉的采样器，评估没有评估语义

```python
# evaluate.py:27-32
# 使用collect_episode_data进行评估
states, actions, rewards, dones, log_probs, values = collect_episode_data(agent, env, action_mask, max_steps=N_STEPS)

# 计算总奖励
total_reward = sum(rewards)
print(f"Total reward accumulated during the evaluation: {total_reward:.3f}")
```

- 只跑**一个回合**（甚至因为环境从空电路开始，只是加 10 个门的固定过程）；
- 奖励是绝对负代价，`total_reward` 随回合长度/电路规模变化，**不是优化效果指标**；
- 没有基线（noop/random/greedy）、没有压缩率、没有等价性检查、没有确定性评估、没有数据集；
- 使用的 `agent` 是 `train.py` 保存的随机模型（见 2.6）。

- **重写版**：`qco/evaluate.py` 在评估集上逐个电路运行 4 种方法，输出 `EvalResult`（门数/深度/代价压缩率、等价率、平均步数、截断率、规则使用分布、耗时），`format_comparison()` 生成 Markdown 表格，`save_eval_results()` 写 `evaluation.json`；`scripts/evaluate.py` 支持 `--checkpoint/--dataset/--out/--baselines`；每个结果都检查 `circuits_equivalent(initial, final)`。

#### 问题 2（P1）：硬编码路径、import 即执行、`exit()`、无种子

模型路径写死 `rl_agent.pth`，文件顶部就创建 agent/env；`exit(1)` 而非抛异常；没有 `argparse`、没有随机种子、没有设备选择。

- **重写版**：`scripts/evaluate.py --help` 可用；所有路径/种子/设备走参数；评估结果写入指定目录。

#### 问题 3（P2）：未使用的导入与过时 API

`Categorical` 未使用；`torch.load` 未显式 `weights_only`（PyTorch 2.6+ 默认值变化会带来兼容性噪音）。

---

### 2.9 `config.py` —— 全局常量、无校验、跨文件硬编码

**文件规模**：13 行。

```python
N_QUBITS = 5
N_MOMENTS = 15
N_GATE_CLASSES = 4     # 门类数量
N_RULES = 6            # 规则数量
N_STEPS = 10
EPOCHS = 5
STEPS_PER_EPOCH = 10
LEARNING_RATE = 5e-4
gamma = 0.99
clip_epsilon = 0.2
```

- **问题 1（P1）**：`N_RULES=6` 必须与 `rules.py` 的 `RULES` 长度手工保持一致；`N_GATE_CLASSES=4` 与 `agent.py` 输入通道、`environment.get_gate_type` 的 4 类硬编码互相耦合；任何一处改动都可能静默错位。
- **问题 2（P1）**：`N_STEPS` 与 `STEPS_PER_EPOCH` 语义重叠（都等于 10），`N_STEPS` 在 `train.py` 里未使用。
- **问题 3（P1）**：没有 seed、device、路径、评估周期、checkpoint、batch size、GAE/熵系数、lambda_depth 等关键配置；`EPOCHS=5` 对 PPO 明显不足。
- **重写版**：`EnvConfig / PPOConfig / TrainConfig` 三个不可变 dataclass（带 `__post_init__` 校验），CLI 覆盖；`RULES` 长度、`N_CHANNELS`、`n_actions` 全部由对象关系推导，不再手工写死。

---

### 2.10 `problem.md` —— 维度争议的“真实结论”

**文件规模**：40 行；是原仓库对策略网络输出维度的一段问题记录。

原文核心论断（`problem.md:17-38`）：

```python
policy = self.policy_linear(x_flat)
policy = policy.view(-1, self.n_rules, self.n_qubits * self.n_moments)
policy = F.softmax(policy, dim=-1)
```

> “这里……实际上您是将每个样本的输出重新整理成一个三维张量……这种处理方式意味着模型为每一条规则在所有量子比特和所有时间步的组合上输出一个概率，这可能会导致策略的解释和应用变得复杂。通常，更直观的方法是让模型输出每个量子比特和每个时间步对每条规则的独立概率，即形状为 `[batch_size, n_qubits, n_moments, n_rules]` 的张量。”

并建议：

```python
policy = policy.view(-1, self.n_qubits, self.n_moments, self.n_rules)
policy = F.softmax(policy, dim=-1)  # 在规则维 softmax
```

**真实结论：问题意识正确，但诊断与建议都不足以修复系统。** 分点说明：

1. **维度顺序本身不是错误。** `[B, R, Q, M]` 与 `[B, Q, M, R]` 只是内存布局不同，乘/加/softmax 的语义只要与动作编码、掩码展平顺序一致就是等价的；网络输出什么顺序完全可以由实现约定。把“不直观”当成 bug 是误判。
2. **真正的错误是归一化范围**：原实现在 `agent.py` 里对 `Q*M` 做 softmax（得到 `R` 个各和为 1 的分布），在 `collect_episode_data.py` 里展平后交给 `Categorical`（对 `R*Q*M` 再归一化），在 `optimizer.py` 里又 `exp`/归一化/softmax（第三次）。**同一个动作在不同阶段对应不同分布**，`old_log_prob` 与 `new_log_prob` 不可比，PPO 的 ratio 没有意义。
3. **采样与 log_prob 的分解必须一致**：原采样在展平联合空间上做，log_prob 却按“先规则、再位置”两段计算；`problem.md` 的建议若用于两阶段决策，仍需要“位置分布 × 条件规则分布”的正确分解，仅把最后一维改成规则并 softmax 并不能保证联合概率正确。
4. **建议版本没有解决掩码、位置丢失、logits 约定等问题**；它甚至可能引入新的归一化错误——若每个 `(q,m)` 的规则分布各自和为 1，则整个动作空间的概率之和为 `Q*M`，不是 1。
5. **重写版的正确做法**：网络只输出 logits `[B, R, Q, M]`；`masked_log_softmax` 在展平后的 `R*Q*M` 上做**恰好一次**归一化；动作用单一整数下标 `index = (r*Q + q)*M + m`；采样、`log_prob`、PPO 重算、掩码全部基于同一套编码与同一分布。`docs/architecture.md` §6/§10/§12 已把这条写成不可协商的接口契约，`tests/test_ppo.py` 断言 `ratio` 初值为 1。
6. `problem.md` 的价值在于**暴露了“概率解释不一致”的现象**，但没有定位到根因，也没有可执行修复与验证；原仓库后来在 `TEST/rebuild_agent.ipynb` 中多次改写策略头（不同 softmax 位置/维度），但始终没有形成“采样—log_prob—PPO 更新”一致的闭环。

---

### 2.11 `TEST/*.ipynb` —— 用调试笔记本代替测试

原仓库 `TEST/` 下有 11 个 notebook：`main.ipynb`、`action_embedding.ipynb`、`optimizer.ipynb`、`rebuild_agent.ipynb`、`rebuild_env.ipynb`、`rebuild_data_generator.ipynb`、`test_agent.ipynb`、`test_data_generator.ipynb`、`test_environment.ipynb`、`test_optimizer.ipynb`、`test_rules.ipynb`。

主要问题：

| 问题 | 证据/后果 |
| --- | --- |
| 复制粘贴模块源码到 notebook 里调试 | 例如 `TEST/test_environment.ipynb` 与 `TEST/test_rules.ipynb` 中原样复制了 `moment_index = min(self.n_moments - 1, len(self.circuit))` 的错误代码；notebook 与根目录模块是两份不同步的副本 |
| 用 `print` 代替断言 | 没有 pytest、没有 CI、没有回归保护；一次修复可能悄悄破坏别处 |
| 保留执行输出与 traceback | `TEST/test_optimizer.ipynb` 中保留了 `compute_log_probs` 的报错 traceback，说明原实现确实跑不通，但问题没有被系统修复 |
| 无随机种子 | 每次运行结果不同，实验不可复现 |
| 版本混乱 | `rebuild_agent.ipynb` 里存在多个不同版本的策略头（softmax 位置、维度顺序各不相同），无法判断哪个是“最终版” |
| 仓库卫生 | `__pycache__/` 被直接放进仓库；没有 `requirements.txt`、`README`、`.gitignore` |

重写版用 `tests/` 下的 pytest 套件替代：门库/电路/模拟器/生成器/规则/掩码/环境/网络/PPO/rollout/训练器/评估/脚本共 16 个测试文件（含 `tests/test_audit.py` 的 55 个独立审计用例），CPU 上整套约 72 秒、全部通过；数据与 Notebook 只保留 `notebooks/quickstart.ipynb` 一个干净的上手示例（不执行、不承载测试职责）。

---

## 3. 横向问题（跨文件）

| # | 问题 | 具体表现 | 重写版机制 |
| --- | --- | --- | --- |
| 1 | **没有“优化语义”定义** | 既没有目标电路/参考酉矩阵，也没有等价性检查；规则在加门 | 所有规则必须通过 `circuits_equivalent`（模全局相位）验证；评估输出 `equivalence_rate` 应恒为 1.0 |
| 2 | **模块间接口不一致** | 状态形状 `[1,C,Q,M]` vs `[B,C,Q,M]` vs `[B,Q,C,M]`；动作 `tuple` vs `int`；step 返回 3 元组；PPO 与采集器的 log_prob 语义不同 | `docs/architecture.md` 单一接口契约；`qco/actions.py` 集中动作编码；`RolloutBatch` 显式形状；测试锁定 |
| 3 | **异常处理反向** | `train.py` 吞异常继续；`collect_episode_data.py` 打印后重抛；`evaluate.py` `exit()` | fail loud；脚本用 `argparse` + 返回码；训练器写日志并保留 checkpoint |
| 4 | **不可复现** | 无 seed、无依赖锁定、数据每次随机、结果不落盘 | `set_seed`、`--seed`、数据集 JSON、配置写入 `config.json`、结果写 `results/` |
| 5 | **没有测试** | 11 个 notebook 手跑，无断言 | 16 个 pytest 文件（237 项，含 55 个独立审计用例）+ 随机 200 电路 × 4 规则的规则不变量测试 + [`docs/audit-report.md`](docs/audit-report.md)（P0=0、P1 已修复） |
| 6 | **没有打包/文档** | 无 `requirements.txt`、无 `README`、无 `.gitignore`、`__pycache__` 入库 | 本重写版补齐 `requirements.txt`、`requirements-dev.txt`、`pyproject.toml`、`.gitignore`、`README.md`、`docs/` |
| 7 | **私有 API 依赖** | `prev_op.gate._exponent` | 重写版只依赖自己定义的门矩阵与自逆门集合，不访问第三方私有属性 |
| 8 | **性能隐患** | `compute_log_probs` 用 Python `for i in range(batch_size)` 逐样本算；卷积主干过大且 BatchNorm 死代码 | 向量化 `log_softmax` + `gather`；主干 3 层 64 通道；CPU 可训练 |
| 9 | **数值稳定性** | 对概率做 `exp`、对概率做 softmax、非法动作概率经 `exp(0)=1` 反而变大 | `NEG_INF=-1e9` 掩码、`masked_entropy` 用 `where` 防止 `0·-inf=nan`；拒绝二次归一化 |
| 10 | **`torch.load` 兼容性** | PyTorch 2.6+ 默认 `weights_only=True`，旧式 checkpoint 需注意 | 只保存 `state_dict`（纯张量），不保存自定义类；加载时显式 `map_location` |

---

## 4. 为什么原仓库“看起来能运行、实际没有训练”

把上面的问题按执行链路串起来，可以得到一条确定性的失败链（均可用原代码行号核对；关键 PyTorch 行为已用最小实验验证）：

1. `train.py` 顶层直接构造 agent/env/PPO；`env.reset()` 得到**空电路**的状态 `[1,4,5,15]`。
2. `collect_episode_data()` 开始循环：策略在 450 个动作上采样；`env.apply_rule()` 选中一条“加门规则”，电路加一个门，奖励变成负数；重复 10 步（环境 `done` 恒为 `False`）。
3. 采集成功返回：`states [10,4,5,15]`、`actions [10,2]`、`values [10]`（`value [1,1]` 靠广播写进去）。注意此时保存的状态与动作已经错位（2.5 问题 1）。
4. `train.py` 的 `check_tensor_format` 通过。
5. 进入 `for step in range(10)`：`states[0::10]` 取出形状 `[4,5,15]` 的**单条 3 维状态**，其余 batch 张量同样只剩 1 条。
6. `PPO.update()` 先做 GAE（长度 1 → 循环不执行，优势全 0），随后 `self.agent(states)` 把 3 维张量送入 `Conv2d` → **`RuntimeError: Expected 4-dimensional input ... but got 3-dimensional input`**。
7. `train.py` 的 `except Exception` 捕获、打印 traceback、继续下一 epoch；5 个 epoch 全都如此。
8. 训练结束仍执行 `torch.save(agent.state_dict(), 'rl_agent.pth')` → **保存的是随机初始化模型**；终端打印 “Training Complete”。
9. `evaluate.py` 加载这个未训练模型，复用同一个采集器，打印“总奖励”（一个无意义的负数）。
10. `data_generator.py` 若要使用，第一次调用就 `AttributeError`；`rules.py` 即便被调用也只会改变电路语义。

因此，原仓库**没有任何一个环节满足“可训练 + 语义保持 + 可度量 + 可复现”**。这不是调参问题，而是需要重写核心闭环。

---

## 5. 重写版改进总览

| 原问题 | 原仓库位置 | 重写版机制 | 保障 |
| --- | --- | --- | --- |
| 规则在加门、改变语义 | `rules.py:6-63` | 4 条等价重写规则，`is_applicable` + 严格降代价不变量 | `tests/test_rules.py`：200 电路 × 4 规则，等价 + cost 严格下降 |
| 状态时隙索引错误、门类丢失 | `environment.py:30-55` | `[14,Q,M]` 张量，显式时隙、control/target/idle 通道 | `tests/test_observation.py`、`tests/test_circuit.py` |
| 动作位置被丢弃 | `environment.py:97-105` | `(r,q,m)` 单一整数编码，环境校验掩码 | `tests/test_actions.py`、`tests/test_environment.py` |
| 奖励是绝对负代价 | `environment.py:64-69` | 势函数塑形 `(C_t-C_{t+1})/C_0`，望远镜性质 | `tests/test_environment.py` 奖励公式 |
| 无终止、无 bootstrap | `environment.py:71-74`、`collect_episode_data.py` | `terminated/truncated` 区分，`next_values` bootstrap | `tests/test_environment.py`、`tests/test_rollout.py` |
| 掩码恒真 | `environment.py:107-118` | `build_action_mask` + masked log-softmax / entropy | `tests/test_masking.py` |
| 策略先 softmax、二次/三次归一化 | `agent.py:59-66`、`optimizer.py:72,134-148` | 网络只输出 logits；全程唯一一次 masked log_softmax | `tests/test_agent.py`、`tests/test_ppo.py` |
| GAE 漏最后一步 | `optimizer.py:97-113` | 完整递推 + bootstrap + 手算参考测试 | `tests/test_ppo.py` |
| 回报/价值按 batch 求和 | `optimizer.py:60-68` | 逐样本 returns/values，展平后 minibatch | `tests/test_ppo.py` |
| 采集状态错位 | `collect_episode_data.py:130-140` | `obs_t → a_t → r_t → obs_{t+1}` 严格对齐并保存 mask | `tests/test_rollout.py` |
| train 切片错误 | `train.py:38-47` | horizon × num_envs 展平 + minibatch 多轮 | `tests/test_trainer.py` |
| 吞异常、保存未训练模型 | `train.py:50-66` | fail loud；产物齐全；周期评估与 best checkpoint | `tests/test_trainer.py` |
| 调用不存在 API | `data_generator.py:14-18` | 用 `random_circuits/save/load` 替代 | `tests/test_generator.py` |
| 评估无指标 | `evaluate.py:27-33` | 4 种方法 + 6 类指标 + 等价率 + JSON/Markdown | `tests/test_evaluate.py` |
| 维度争议误诊 | `problem.md` | 单一联合动作分布，编码集中管理 | `tests/test_ppo.py` ratio==1 断言 |
| 无测试/不可复现 | `TEST/*.ipynb`、全局常量 | pytest 套件 + seed + CSV/JSON 落盘 | 全部测试 + `tests/conftest.py` |

---

## 6. 原仓库中仍值得保留的想法

重写版不是“另起炉灶”，原仓库的以下思路被保留并修正：

1. **把电路优化建模为 MDP**：用强化学习做电路重写是合理的选题方向，重写版沿用；
2. **动作 = (规则, 位置)**：原仓库的 `(rule_index, qubit_moment_index)` 二元组思想保留，扩展为 `(rule, qubit, moment)` 并统一编码；
3. **CNN 处理“门-比特-时隙”张量**：保留“用卷积捕捉局部结构”的思路，重写为 14 通道输入、3 层主干、双头输出；
4. **PPO + 价值网络 + 熵正则**：保留 on-policy 算法框架，但把数学实现改正；
5. **`problem.md` 对“概率解释”的质疑**：保留问题意识，重写成正确的“logits → 单一 masked log_softmax”方案。

---

## 7. 已知局限（重写版仍存在的边界，答辩时如实说明）

1. **规则集合不完备**：只覆盖“相邻单比特门合并、自逆二比特门抵消、独占时隙门移动”4 类；不处理参数化旋转门（RZ/RX 角度合并）、三比特门、测量/重置、任意酉矩阵合成等。
2. **随机稠密电路上部分规则适用性低**：默认随机电路每层几乎占满所有比特，`move_earlier/move_later` 要求“门独占一个时隙”，因此触发较少；实验中的深度压缩率可能明显低于门数压缩率（见 `docs/experiment.md` 的讨论模板）。
3. **观测窗口固定**：`n_moments=15` 必须不小于电路深度；扩大到更深电路需要同步增大窗口与网络规模。
4. **泛化范围有限**：策略在固定 `n_qubits=5`、`initial_depth=12` 的训练分布上训练；改变比特数/深度分布需要重新训练（或做结构适配）。
5. **不模拟硬件噪声与连通性约束**：拓扑只假设线性相邻 `(q, q+1)` 可放二比特门，未建模真实设备耦合图/错误率。
6. **未与工业级转译器（Qiskit/TKet/Cirq optimizers）做全面对比**：基线是 noop/random/greedy，足以验证学习是否有效，但不是 SOTA 对比。

---

## 附录 A：原代码引用索引

| 主题 | 文件:行号 |
| --- | --- |
| 策略/价值头与 softmax | `agent.py:24-28, 36-39, 59-66` |
| 状态张量时隙错误 | `environment.py:30-42` |
| 门类型识别 | `environment.py:44-55` |
| 奖励/终止 | `environment.py:57-74` |
| ActionMask 恒真 | `environment.py:107-118` |
| 加门规则 | `rules.py:6-32` |
| 盲目 commute | `rules.py:34-40` |
| 跨比特 cancel | `rules.py:42-61` |
| GAE 漏步 | `optimizer.py:93-113` |
| 二次/三次归一化 | `optimizer.py:72, 115-151` |
| 回报/价值求和 | `optimizer.py:60-68` |
| 单步更新 | `optimizer.py:79-91` |
| 采样与 log_prob 不一致 | `collect_episode_data.py:44-65, 110-140` |
| 状态错位 | `collect_episode_data.py:130-140` |
| 错误切片 | `train.py:38-47` |
| 吞异常/保存模型 | `train.py:50-66` |
| 不存在的 API | `data_generator.py:14-18` |
| 评估只打印总奖励 | `evaluate.py:27-33` |
| 全局配置 | `config.py:1-13` |
| 维度争议 | `problem.md:17-40` |

## 附录 B：审计方法与可复现性说明

- 本分析基于对原仓库源码的静态阅读；所有行号以原仓库当前文件为准；
- 关键平台行为（PyTorch 广播赋值、`Conv2d` 要求 4 维输入、`t[i::n]` 切片、空张量均值）在重写版的 `.venv` 中做过最小验证，确认失败链描述成立；
- 原仓库未被执行/修改，避免产生 `rl_agent.pth` 等副作用；
- 重写版每个修复点都有对应的 pytest 用例，可用 `.venv\Scripts\python.exe -m pytest tests -q` 验证；规则等价性与 GAE 另有独立审计的对抗性测试复核（`tests/test_audit.py`，报告见 [`docs/audit-report.md`](docs/audit-report.md)）。
