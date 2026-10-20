# 方法设计（docs/design.md）

> 本文是重写版毕业设计的方法说明，回答“系统是怎么设计的、为什么这样设计、正确性如何保证”。
> 实现接口以 [`docs/architecture.md`](architecture.md) 为唯一契约；README 只做概览，
> 答辩速记见 [`docs/defense-notes.md`](defense-notes.md)，实验细节见 [`docs/experiment.md`](experiment.md)。

---

## 1. 设计目标与三条硬不变量

### 1.1 目标

给定一个由离散量子门组成的电路 `C`，在不改变其计算语义（幺正矩阵模全局相位）的前提下，
通过一系列**等价重写**降低电路代价：

```
cost(C) = num_ops(C) + λ · depth(C),   默认 λ = 0.5
```

其中 `num_ops` 是门的总数，`depth` 是电路的时隙（layer / moment）层数。λ 体现“深度比门数更贵”
的硬件直觉：深度直接对应电路执行时间，门数对应操作数与错误累积。

### 1.2 三条硬不变量（贯穿全部模块）

| 编号 | 不变量 | 保障机制 |
| --- | --- | --- |
| **I1 语义保持** | 任一允许执行的重写动作，应用前后电路的幺正矩阵**模全局相位**相等 | 规则实现 + `qco/simulator.circuits_equivalent`；`tests/test_rules.py` 对 200 个随机电路 × 4 规则验证；评估等价率必须为 1.0 |
| **I2 严格进步** | 任一可行动作应用后 `cost` 严格下降 | 每条规则只做“删门 / 合并门 / 删空层”；环境 `step` 内用断言防御（不变量被破坏即崩溃） |
| **I3 动作合法** | 策略只在当前可行动作上采样；非法动作概率严格为 0 | `build_action_mask` + `masked_log_softmax`；环境 `step` 对非法动作抛 `ValueError` |

这三条不变量把“强化学习可能学坏”的风险限制在**策略选择**层面：无论策略多差，电路永远是合法的、
语义永远不变、每一步都不会让代价变差。这是原仓库最缺失的部分（它的规则在往电路上加门，见
[`PROBLEM_ANALYSIS.md`](../PROBLEM_ANALYSIS.md)）。

---

## 2. 问题形式化

### 2.1 电路表示

- 电路 `C = (n, L_0, ..., L_{D-1})`，`n` 为比特数，`D` 为深度，每个 `L_t` 是一个时隙（moment）；
- 第 `t` 层中不同门作用的比特集合互不相交，因此同一层内的门彼此对易；
- 门的类型固定为 11 种离散门（`X,Y,Z,H,S,Sdag,T,Tdag,CNOT,CZ,SWAP`），
  不含连续角度参数——这是“动作空间可离散化”的前提；
- 所有电路均不可变（`tuple of tuple`），任何重写返回新对象。

### 2.2 优化问题

```
minimize   cost(C_T)
subject to U(C_T) = e^{iφ} · U(C_0)   （模全局相位等价）
           C_T 由 C_0 经一系列规则应用得到
```

这是一个**带语义约束的组合优化**问题：搜索空间是“所有与初始电路等价的电路”，目标是代价最小。
精确求解需要穷举/等价类搜索，代价指数级；本项目用强化学习学习一个**启发式策略**来近似求解。

---

## 3. MDP 形式化

### 3.1 状态（state）

状态是当前电路的**门-比特-时隙张量**：

```
s_t ∈ R^{C × Q × M},   C = 14, Q = n_qubits, M = n_moments
```

- 通道 0–10：11 种门的 one-hot 标记（二比特门在两个比特上同时置 1）；
- 通道 11（`CONTROL_CHANNEL`）：二比特门的第一个比特（按 `Op.qubits` 顺序，即 control）；
- 通道 12（`TARGET_CHANNEL`）：二比特门的第二个比特（target）；
- 通道 13（`IDLE_CHANNEL`）：该比特在该时隙空闲；
- 时隙下标 ≥ 当前电路深度时，所有比特都标记为空闲（padding），保证形状固定；
- 状态是电路的**无损编码**：给定观测（`M ≥ D`）可唯一还原电路的门与时隙（二比特门通过
  control/target 通道成对还原）。

为什么不用态矢量/密度矩阵做状态：重写规则只关心“调度结构”，不需要复数振幅；
用 one-hot 张量让 CNN 的输入尺度稳定、训练简单，也避免把物理态的计算开销带进每一步决策。

### 3.2 动作（action）

```
a_t = (r, q, m),   r ∈ {0,1,2,3},  q ∈ [0,Q),  m ∈ [0,M)
```

规则下标固定（禁止调换）：

| r | 规则 | 语义 |
| --- | --- | --- |
| 0 | `merge_single_qubit` | 同一比特相邻两个单比特门相乘，结果命中门库则合并，等于 `I` 则删除两个门 |
| 1 | `cancel_two_qubit` | 同一有序比特对上相邻的两个相同自逆二比特门（`CNOT/CZ/SWAP`）抵消 |
| 2 | `move_earlier` | 独占时隙的门前移到最早可放置位置，深度 −1 |
| 3 | `move_later` | 独占时隙的门后移到最晚可放置位置，深度 −1 |

动作空间大小 `|A| = R·Q·M`，默认配置下为 `4 × 5 × 15 = 300`。

动作是**规则宏**，不是“往某处放某个门”。这样做的理由：

1. **语义安全**：每个宏都经过等价性证明与测试，策略不做“拼接门”的探索；
2. **样本效率**：一次动作可能同时删 1–2 个门或压缩 1 层深度，搜索空间远小于原语动作；
3. **可解释**：规则使用次数可直接统计，答辩时能说清“智能体学会了什么”。

代价是规则集合不完备（见第 12 节），这是本设计明确承认的取舍。

### 3.3 转移（transition）

环境是**确定性**的：`s_{t+1} = f(s_t, a_t)`，`f` 由规则 `apply` 给出；唯一的随机性来自
初始电路采样与策略采样。动作合法性由掩码定义：

```
mask[r, q, m] = is_applicable_r(C, q, m) ∧ (m < depth(C))
```

被掩码屏蔽的动作不可采样；环境 `step` 对非法动作直接抛异常，防止“实现 bug 被静默学进去”。

### 3.4 奖励（reward）：势函数塑形

记 `C_t = cost(C_t)`：

```
r_t = (C_t - C_{t+1}) / C_0
```

其中 `C_0` 是回合初始电路的代价。性质：

1. **密集**：只要执行了动作（合法动作必定降低代价），奖励就为正；
2. **有界**：`0 < r_t ≤ 2/C_0`（一次合并最多删 2 个门，移动最多减少 λ 的深度项），
   数值范围与电路规模无关；
3. **可解释**：按回合未折扣求和，总回报恰好是“相对初始代价的压缩比例”：

   ```
   Σ_{t=0}^{T-1} r_t = (C_0 - C_T) / C_0
   ```

这属于**势函数塑形**（potential-based shaping）的离散形式：`r_t = Φ(s_t) − Φ(s_{t+1})`，
`Φ(s) = C(s)/C_0`。需要说明的是，PPO 优化的是折扣回报 `Σ γ^t r_t`，严格来说 γ<1 时
标准势函数塑形形式应为 `γΦ(s') − Φ(s)`；本设计选择“未折扣势差 + γ=0.99 + 回合长度 ≤ 48”
的组合，原因是：

- 每步奖励同号、回合短，折扣对最终排序的影响很小；
- 未折扣形式让奖励有清晰的物理含义（压缩比例），答辩解释成本低；
- 评估指标直接看最终代价压缩率，不依赖回报数值，因此奖励塑形的细微差异不影响主结论。

这在 [`docs/defense-notes.md`](defense-notes.md) 中有对应的问答准备（“为什么不用 −cost 做奖励”）。

### 3.5 终止（termination）

```
done = terminated ∨ truncated
terminated = 当前电路不存在任何可行动作
truncated  = steps ≥ max_steps （默认 48）
```

区分终止/截断是 GAE bootstrap 正确性的前提：真正终止时 `next_value = 0`；
截断时仍用 `V(s_T)` 做 bootstrap。`info` 中同时返回 `terminated` / `truncated`。

### 3.6 回合（episode）

1. 从随机初始电路分布采样 `C_0`（5 比特、深度 12、二比特门概率 `p_two=0.15`）；
2. 反复执行 `(r,q,m)`，直到无动作可做或达到 `max_steps`；
3. 回合数据 `(s_t, a_t, logπ, V(s_t), r_t, done_t, mask_t)` 交给 PPO。

---

## 4. 门库与电路表示

### 4.1 门库（`qco/gates.py`）

- 门顺序固定：`X, Y, Z, H, S, Sdag, T, Tdag, CNOT, CZ, SWAP`，该顺序即观测张量通道顺序；
- 单比特门为 2×2 幺正矩阵，二比特门为 4×4；比特 0 为最高位（big-endian），
  基矢顺序 `|00...0>, |00...1>, ..., |11...1>`；
- `Sdag = diag(1, −i)`、`Tdag = diag(1, e^{−iπ/4})`；
- `"I"` 是特殊名字，仅表示“合并结果为单位阵”，不是可放置的门；
- `match_single_qubit_gate(U)` 在**模全局相位**意义下匹配门库：若 `U = e^{iφ}G` 则命中 `G`，
  若 `U = e^{iφ}I` 返回 `"I"`，否则 `None`。实现方式：取 `U` 与目标矩阵绝对值最大的元素作为
  相位基准，比较 `U ≈ phase·G`。

### 4.2 电路（`qco/circuit.py`）

- `Circuit(n_qubits, layers)` 构造时校验：比特下标合法、同一层内比特不冲突、
  多比特门比特不重复、自动丢弃空层；
- `layers` 与 `Op` 均不可变；`Op(gate, qubits)` 中 `CNOT` 的顺序是 `(control, target)`；
- 提供 `depth / num_ops / cost / op_on / next_op_on / prev_op_on / layer_qubits` 等查询，
  规则实现只依赖这些接口；
- `unitary` 是延迟计算的 `cached_property`；`equivalent()` 调用模拟器；
- `to_dict/from_dict` 用于数据集 JSON；`to_cirq/from_cirq` 为可选互操作。

不可变设计的收益：规则实现不会意外改到原电路；电路可做字典键（便于去重/搜索）；
单元测试可以拿“输入/输出”直接比较；并行采样时多个环境可以共享同一电路对象。

---

## 5. 四条重写规则与正确性论证

记 `U(C)` 为电路 `C` 的幺正矩阵，`≅` 表示模全局相位相等。每条规则都满足：

> **定理（规则正确性）**：若 `rule.is_applicable(C, q, m)` 为真，则
> `C' = rule.apply(C, q, m)` 合法、`U(C') ≅ U(C)`，且 `cost(C') < cost(C)`。

### 5.1 `MergeSingleQubitRule`（r=0）

**适用条件**：`q` 在时隙 `m` 上有单比特门 `G1`；沿 `q` 向后找到下一个门 `G2`（中间没有
任何门作用于 `q`）；`G2` 也是单比特门；`U = G2·G1` 在模全局相位意义下命中门库或等于 `I`。

**变换**：若 `U ≅ I`，删除两个门；若 `U ≅ G`，把 `G` 放在原 `G1` 的时隙、删除 `G2`。

**正确性**：`G1` 与 `G2` 之间作用于其他比特的门与 `G1,G2` 张量积对易（不相交子系统），
所以 `U(C) = (...)·(G2 G1)_q·(...)`；用 `e^{iφ}G` 替换 `G2G1` 只引入全局相位。
若为 `I`，删除两个门等于乘单位阵。

**代价**：门数 `−1`（合并）或 `−2`（删除），深度不增；`cost` 严格下降。

**例**：`T·T ≅ S`、`H·H ≅ I`、`Z·Y ≅ X`。

### 5.2 `CancelTwoQubitRule`（r=1）

**适用条件**：`(q, m)` 上的门 `G` 是 `CNOT/CZ/SWAP` 之一；`G` 的两个比特 `a,b` 各自的
下一个门存在且**是同一时隙、同一个门**（即两个比特之间的下一个门就是彼此的配对 `G`）。

**变换**：删除这两个相同的二比特门。

**正确性**：`CNOT/CZ/SWAP` 都是自逆门（`G² = I`）；由于两个比特上的下一个门都是 `G` 本身，
两者之间没有门触及 `a` 或 `b`，其他比特上的门与 `G` 对易，因此 `...G...G... = ...I...`。

**代价**：门数 `−2`；深度不增；严格下降。

### 5.3 `MoveEarlierRule`（r=2）

**适用条件**：时隙 `m` 只有 `op` 一个门（`len(layer(m)) == 1`）；向 `m−1` 方向扫描，
跳过所有与 `op.qubits` 不相交的层，得到可放置的最早位置 `p < m`。

**变换**：删除时隙 `m` 的 `op`，放入时隙 `p`；其余层顺序不变，空层自动丢弃。

**正确性**：`op` 只跨过“不触及它作用比特的层”，这些层上的门与 `op` 对易；每个比特上
门的相对顺序不变（因为所跨层上没有该比特的门），所以整条电路的酉矩阵不变。
**深度**：原时隙 `m` 在删除后为空层并被丢弃，深度 `−1`。

**代价**：门数不变，深度 `−1`；严格下降。

### 5.4 `MoveLaterRule`（r=3）

与 5.3 对称，向 `m+1` 方向扫描到最晚可放置位置 `p > m`；正确性论证相同，深度 `−1`。

> **为什么要求“独占时隙”**：如果 `op` 所在层还有其他门，删除 `op` 后该层不空、深度不降，
> 只有门数不变而深度不变——违反“严格进步”不变量。要求独占层保证每次移动都必然删掉一层。

### 5.5 不变量 I2（严格进步）的总证明

| 规则 | 门数变化 | 深度变化 | cost 变化 |
| --- | --- | --- | --- |
| merge（合并/删除） | −1 或 −2 | 0 或更少 | `≤ −1` |
| cancel | −2 | 0 | `−2` |
| move_earlier / move_later | 0 | −1 | `−λ = −0.5` |

所以任一可行动作的 `cost` 至少下降 `min(1, 2, λ) = 0.5`（合并删除单位阵时也是 −2；合并门数 −1）。
结合 `cost ≥ 0`，回合长度天然有上界：`T ≤ cost(C_0)/0.5`（远大于 `max_steps=48`，
因此实际很少因代价下界而终止）。

### 5.6 掩码与“可行动作”的一致性

```
mask[r,q,m] = RULES[r].is_applicable(C, q, m) ∧ m < depth(C)
```

`applicable_actions(C)` 枚举同一表达式；`tests/test_masking.py` 交叉验证二者一致，
并断言被屏蔽位置的策略概率严格为 0。

---

## 6. 动作编码（`qco/actions.py`）

固定布局：

```
index = (rule * n_qubits + qubit) * n_moments + moment
```

- 与策略网络输出 `[B, R, Q, M]` 的展平顺序、掩码 `[R,Q,M]` 的展平顺序完全一致；
- 编码/解码集中在 `qco/actions.py`，避免“网络按 `[Q,M,R]` 展平、掩码按 `[R,Q,M]` 展平”这类
  静默错位；
- `encode_actions/decode_actions` 提供向量化版本，方便分析可行动作集合。

中心化编码是原 `problem.md` 维度争议的工程解药：只要**唯一**决定布局，
张量维度顺序就只是一个约定，不存在“对/错”之争（详见 `PROBLEM_ANALYSIS.md` §2.10）。

---

## 7. 掩码机制（`qco/masking.py`）

### 7.1 为什么必须有掩码

动作空间 300 个，而默认稠密随机电路上可行动作通常只有十几个到几十个（一次合并/抵消/移动）。
若不掩码：

- 大量概率质量被浪费在“必然非法”的动作上，探索效率极低；
- 采样到非法动作后只能“重采样/随机动作”，`log_prob` 与实际采样分布不一致，PPO 的 ratio 失去意义；
- 智能体可能学到“乱试”，而不是“在可行动作中选最优”。

### 7.2 实现要点

```python
NEG_INF = -1e9

def masked_log_softmax(logits, mask):
    masked_logits = logits.masked_fill(~mask, NEG_INF)
    return torch.log_softmax(masked_logits, dim=-1)
```

- **一次归一化**：只在 `masked_log_softmax` 内做一次，网络与 PPO 都不再做 softmax；
- **非法动作严格 0 概率**：`NEG_INF = -1e9`（不是 `-inf`），避免 `log_softmax` 出现 `-inf` 后
  在 `exp`/乘法中产生 `nan`，同时 `exp(-1e9 - max) = 0` 在 float32 下严格为 0；
- **熵只算有效动作**：

  ```python
  probs = where(mask, exp(log_probs), 0)
  terms = where(mask, probs * log_probs, 0)
  entropy = -terms.sum(-1)
  ```

  直接 `0 · (-inf)` 会得到 `nan`，因此必须用 `where` 屏蔽；
- **PPO 重算时必须使用同一组 mask**：`RolloutBatch.masks` 随数据一起保存，
  `PPO.update` 重算 `log_prob` 时复用，确保 `old_log_prob` 与 `new_log_prob` 是同一分布的两次取值。

### 7.3 一个回合内掩码的变化

每次 `step` 后电路改变，掩码重新计算（`env.action_mask` 惰性缓存、step 后失效）。
回合终止条件正是“掩码全 False”，因此策略永远不会遇到“无动作可采”的困境。

---

## 8. 策略网络（`qco/agent.py`）

### 8.1 结构

```
观测 [B, 14, Q, M]（float32）
        │
        ├─ Conv2d(14, H, 3, padding=1) + GroupNorm + ELU
        ├─ Conv2d(H,  H, 3, padding=1) + GroupNorm + ELU
        ├─ Conv2d(H,  H, 3, padding=1) + GroupNorm + ELU   # 共享主干，H = hidden_channels = 64
        │
        ├─ 策略头: Conv2d(H, R, 1)                → logits [B, R, Q, M]
        │
        └─ 价值头: Conv2d(H, 32, 1) + ELU
                   → Flatten → Linear(32·Q·M, 128) + ELU
                   → Linear(128, 1) → value [B]
```

- **不变量**：`forward()` 返回 `(logits [B,R,Q,M], value [B])`，**绝不在网络内做 softmax**；
- **不使用 BatchNorm**：PPO 的采集/更新分布持续变化，小批量下 BN 的滑动统计量会引入偏差，
  且训练/推理模式行为不一致；改用 **GroupNorm**（与批大小无关，分组数自动取 8/4/2/1 中能整除
  通道数的第一个）+ ELU，在默认配置下稳定训练；
- 卷积主干天然利用“比特相邻、时隙相邻”的局部性；3 层 3×3 卷积的感受野为 7×7，
  覆盖 5 个比特的全部（`Q=5`），但只覆盖 15 个时隙中的 7 个——这是策略头的局部视野限制，
  见第 12 节；
- 默认 `H=64` 时全网络共 **392,165** 个参数（`sum(p.numel() for p in net.parameters())`），
  CPU 上单次前向为毫秒级。

### 8.2 `act()` 的语义

```python
def act(observations, masks, deterministic=False):
    logits, value = self.forward(observations)         # [B,R,Q,M], [B]
    flat_masks = masks.reshape(B, -1) if masks.dim() > 1 else masks.expand(B, -1)
    log_probs = masked_log_softmax(logits.reshape(B, -1), flat_masks)  # [B, R*Q*M]
    dist = Categorical(log_probs=log_probs)
    action = masked_logits.argmax(-1) if deterministic else dist.sample()
    return action, log_probs.gather(-1, action), value, masked_entropy(log_probs, flat_masks)
```

- 采样与 `log_prob` 来自**同一个** `log_probs`，从根源上避免原仓库“采样分布 ≠ 重算分布”的问题；
- `deterministic=True` 用于评估：对掩码后的 logits 取 `argmax`（与对 `log_probs` 取 `argmax` 等价，
  实现上更直接）；
- 熵在有效动作上计算，作为 PPO 的探索项；
- 若批次中某行掩码全为 `False`（理论上环境已终止），`act` 会抛 `ValueError`，
  采样器在回合结束时会先 auto-reset，保证调用 `act` 时总有可采动作。

### 8.3 为什么是 Actor-Critic 双头

- 共享主干：策略与价值共享“电路结构特征”，样本效率高于两套独立网络；
- 价值头用于 GAE 的优势估计，降低策略梯度方差；
- 双头输出让一次前向同时得到动作分布与状态价值，rollout 采样成本低。

---

## 9. PPO 算法与实现要点（`qco/ppo.py`）

### 9.1 目标函数

对每个样本 `(s, a, A, R, mask)`，记新旧策略比：

```
ρ_t = π_θ(a_t | s_t) / π_θold(a_t | s_t) = exp(logπ_new − logπ_old)
```

裁剪替代目标：

```
L^CLIP = E[ min( ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t ) ]
```

总损失：

```
L = −L^CLIP + c_v · MSE(V_θ(s_t), R_t) − c_e · H[π_θ(·|s_t)]
```

默认超参（`PPOConfig`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `lr` | 3e-4 | Adam 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 的 λ |
| `clip_epsilon` | 0.2 | PPO 裁剪范围 |
| `value_coef` | 0.5 | 价值损失权重 |
| `entropy_coef` | 0.01 | 熵正则权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪 |
| `update_epochs` | 4 | 每批数据重复利用轮数 |
| `minibatch_size` | 64 | 小批量大小 |
| `target_kl` | 0.02 | KL 早停阈值 |
| `normalize_advantage` | True | 是否归一化优势 |

### 9.2 GAE（广义优势估计）

逐环境、反向递推：

```
δ_t = r_t + γ · V(s_{t+1}) · (1 − d_t) − V(s_t)
A_t = δ_t + γ·λ · (1 − d_t) · A_{t+1}
```

- `V(s_{t+1})` 在 `t` 为最后一步时取 `next_values[env]`（bootstrap 价值）；
- 递推**必须包含最后一步**——原仓库 `range(len(rewards)-1)` 漏掉最后一步，导致最后一步优势恒 0、
  回报恒等于价值，这是本项目在 `PROBLEM_ANALYSIS.md` 中重点修复的缺陷；
- `returns = advantages + values`（价值头拟合目标）；
- 用 `(1 − d_t)` 而不是“遇 done 就清零”是为了正确处理截断与 auto-reset：回合结束时价值截断，
  新回合的初始观测只作为 bootstrap，不污染上一条轨迹。

### 9.3 更新流程

```python
advantages, returns = compute_gae(rewards, values, dones, next_values, γ, λ)
# 展平 [E, T] → [E*T]
for epoch in range(update_epochs):
    if target_kl reached: break
    for minibatch in shuffle(flat_samples, minibatch_size):
        logits, value = policy(obs)
        log_probs = masked_log_softmax(logits, masks)        # 复用同一 mask
        ratio = exp(log_probs[action] - old_log_probs)
        surr1, surr2 = ratio * adv, clamp(ratio, 1±ε) * adv
        policy_loss = -min(surr1, surr2).mean()
        value_loss = mse(value, returns[action])
        entropy = masked_entropy(log_probs, masks).mean()
        loss = policy_loss + c_v*value_loss - c_e*entropy
        loss.backward(); clip_grad_norm_; optimizer.step()
```

输出指标：`policy_loss / value_loss / entropy / approx_kl / clip_fraction / total_loss /
explained_variance / learning_rate / grad_norm`，用于判断训练是否健康。

### 9.4 数值与工程细节

- **不允许双重 softmax**：网络输出 logits，`masked_log_softmax` 是唯一归一化入口；
- **优势归一化**：展平后按批归一化（`normalize_advantage`），小批量单样本时跳过；
- **KL 早停**：用 `approx_kl` 估计策略漂移，超过 `target_kl` 立即停止后续 epoch，
  防止策略在单批数据上过度更新；实现中 `approx_kl = mean(logπ_old − logπ_new)`，
  即 `E[log ρ]` 的一阶估计（小步长下约等于 `E[(ρ−1) − log ρ]` 的 KL 估计）；
- **梯度裁剪**：`max_grad_norm=0.5`，防止 PPO 训练偶发的大梯度破坏网络；
- **batch 结构**：`[num_envs, horizon]` 展平为 `num_envs × horizon` 条样本，
  再切 minibatch；原仓库 `states[step::STEPS_PER_EPOCH]` 的切片错误在这里被彻底消除。

---

## 10. 采样与训练流程（`qco/rollout.py`、`qco/trainer.py`）

### 10.1 RolloutCollector

- 输入：`num_envs` 个环境、horizon 步、`circuit_sampler`；
- 每步：对 `num_envs` 个环境组成的 batch 做一次前向，按掩码采样动作，记录
  `(obs, action, log_prob, value, reward, done, mask)`；
- 某环境 `done` 后：记录 `dones=1`，用 `circuit_sampler()` 采样新电路并 `reset`，
  下一步进入新回合；`next_values[env]` 取“最后一步之后的观测”（新回合初始观测）的价值；
- 产物：`RolloutBatch`（observations/actions/log_probs/values/rewards/dones/masks/next_values）
  与 `EpisodeStats` 列表（回报、初始/最终代价、门数、深度、步数、截断标记、规则使用次数）。

### 10.2 Trainer

一个训练周期 = `collect → update`；默认：

| 配置 | 值 |
| --- | --- |
| `iterations` | 150（默认；本报告正式实验使用 200） |
| `steps_per_iter`（horizon） | 16 |
| `num_envs` | 8 |
| `hidden_channels` | 64 |
| `eval_every` | 10 |
| `save_every` | 25 |
| `train_circuits` | 200（seed=0 生成） |
| `eval_circuits` | 100（seed=10000，或加载 `data/eval.json`） |
| `device` | cpu |

产物（写入 `--out` 目录）：

- `config.json`：完整配置（含 `EnvConfig/PPOConfig/TrainConfig`）；
- `train_log.csv`：每轮 `iteration, mean_return, mean_episode_len, mean_initial_cost,
  mean_final_cost, gate_reduction_pct, depth_reduction_pct, policy_loss, value_loss,
  entropy, approx_kl, explained_variance`；
- `eval_log.csv`：每次评估的 `method, cost_reduction_pct, gate_reduction_pct,
  depth_reduction_pct, equivalence_rate, mean_steps`；
- `best.pt` / `last.pt`：按评估集 `mean_final_cost` 最优保存的 checkpoint；
  另按 `save_every`（默认 25）保存周期性检查点 `checkpoint_XXXX.pt`；
- `summary.json`：最终 noop/random/greedy/agent 对比结果（含 `comparison_table`、训练耗时、
  最优轮次等）。

### 10.3 评估（`qco/evaluate.py`）

对评估集每个电路分别运行 4 种方法：

| 方法 | 行为 |
| --- | --- |
| `noop` | 不动作（下界基准） |
| `random` | 从可行动作中随机选（探索基准） |
| `greedy` | 每步枚举可行动作，选“执行后 cost 最小”的动作（一步贪心基线） |
| `agent` | 策略网络 `deterministic=True`（评估时取最大概率动作） |

每个结果都检查 `circuits_equivalent(initial, final)`；聚合输出 6 类指标（定义见
[`docs/experiment.md`](experiment.md)）。`format_comparison()` 输出 Markdown 表
（方法、初始→最终门数/深度、三类压缩率、等价率、平均步数）；截断率与平均耗时在
`EvalResult`/JSON 中提供。

---

## 11. 复杂度分析

设比特数 `Q`、时隙窗口 `M`、规则数 `R=4`、电路深度 `D ≤ M`、门数 `N`。

| 阶段 | 时间复杂度 | 空间复杂度 | 默认配置下量级 |
| --- | --- | --- | --- |
| 观测编码 | `O(C·Q·M + N)` | `O(C·Q·M)` | 1050 个 float32 ≈ 4.2 KB |
| 掩码构造 | `O(R·Q·D · 单规则判定)`，单规则判定最坏 `O(D·Q)` | `O(R·Q·M)` | 300 个 bool |
| 单步 `step` | 规则复制 `O(D·Q)` + 观测/掩码 | `O(C·Q·M)` | 微秒级 |
| rollouter 一轮 | `O(E·T·(mask+obs+forward))` | `O(E·T·C·Q·M)` | `8×16×14×5×15×4 B ≈ 0.54 MB` |
| PPO 更新 | `O(epochs · E·T · forward)` | 同 batch | 128 样本 × 4 epoch，CPU 毫秒级 |
| 幺正矩阵 | `O(N · 4^Q)`（一次性乘全部基矢） | `O(4^Q)` | `Q=5` 时 32×32 矩阵 |
| 等价性校验 | 两电路各 `O(N·4^Q)` | 同上 | 评估 100 个电路仍 < 1 s |

网络前向复杂度：三层 3×3 卷积 `O(H²·Q·M)`，策略头 `O(H·R·Q·M)`，价值头
`O(H·32·Q·M + 32·Q·M·128)`；`H=64`、`Q=5`、`M=15` 时单样本约 10⁶ 次乘加，CPU 足够。

训练总耗时与机器相关：本次正式实验（200 轮、CPU i5-1035G1）实测 **986.17 s（约 16.4 分钟）**，
平均 4.93 s/轮；独立评估 100 个电路耗时约 19 s（agent，190.34 ms/电路）。详见
[`docs/experiment.md`](experiment.md) 的“效率”一节。

---

## 12. 与 Cirq 的关系

### 12.1 定位：可选互操作，不是运行时依赖

- 核心环境、规则、模拟器、网络、训练、评估**全部只依赖 numpy/torch**；
- 安装 cirq 后获得两项附加能力：
  1. `Circuit.to_cirq()` / `from_cirq()`：与 Cirq 生态交换电路；
  2. `Circuit.to_text()`：用 Cirq 的电路图渲染结果做展示；
- 未安装 cirq 时：`to_cirq/from_cirq` 抛带安装提示的 `RuntimeError`，`to_text()` 回退到自绘 ASCII 图；
- 测试中若检测到 cirq，则用 `cirq.unitary(circuit.to_cirq())` 与 `qco/simulator.py` 的结果
  在模全局相位意义下交叉验证（`tests/test_circuit.py`、`tests/test_simulator.py`，无 cirq 时 `skip`）。

### 12.2 基矢约定

本项目与 Cirq 的基矢顺序一致：比特 0 为最高位，`qubits` 顺序与门矩阵的行列顺序一致；
对本项目门库中的门，`qco` 的幺正矩阵与 `cirq.unitary(to_cirq())` 逐元素相等
（测试用“模全局相位相等”断言更稳健，允许未来门集扩展到带相位约定的门）。
`from_cirq` 按 `sorted(circuit.all_qubits())` 的次序给比特编号，因此只支持“连续从 0 开始编号”的
Cirq 电路（与 `n_qubits` 一致）。

### 12.3 为什么不用 Cirq 直接做环境

1. **可控性**：本项目需要精确控制“时隙/层”语义、全局相位判定与门库顺序，自己实现模拟器更直接；
2. **依赖稳定性**：Cirq 版本迭代会影响门类 API（原仓库就用了 `_exponent` 私有属性）；核心逻辑
   不依赖它可以让项目在任何 Python 3.10+ 环境复现；
3. **可测试性**：numpy 模拟器代码短、可读，便于在论文中解释与在答辩中现场演示；
4. **可验证性**：Cirq 仍作为“第三方裁判”参与交叉验证，兼顾独立性与自主可控。

---

## 13. 设计取舍与替代方案

| 设计选择 | 备选方案 | 为什么这样选 |
| --- | --- | --- |
| 规则宏作为动作 | 逐门合成/直接操作门序列 | 保证语义安全与每步进步；动作空间小（300）可训练；原仓库的失败正说明“无约束动作”不可行 |
| 确定性环境 | 随机转移 | 重写规则本身确定；随机性只来自初始电路与策略，便于复现与调试 |
| 势函数奖励 | 终止时一次性给 `−cost` | 稀疏奖励在 48 步回合内极难学习；塑形奖励密集、有界、可解释 |
| PPO | DQN / A2C / SAC | 动作空间离散且巨大（300），DQN 需要 max over actions；PPO 裁剪稳定、实现简单、on-policy 与掩码兼容；连续控制类算法（SAC/DDPG）不适用 |
| CNN Actor-Critic | MLP / Transformer / GNN | 电路是“比特×时隙”的二维结构，CNN 参数共享且对局部模式敏感；MLP 参数爆炸；GNN 更灵活但实现复杂，留作展望 |
| 不可变电路 | 原地修改 | 规则实现简单、可哈希、可并行、测试友好；原仓库原地修改与状态不同步是 bug 来源之一 |
| 单一联合动作编码 | 两阶段（先规则后位置） | 联合分布才能让采样与 log_prob 一致；两阶段需要严格的概率分解，容易重蹈原仓库覆辙 |
| numpy 模拟器 | Cirq 模拟器 | 可控、轻量、可测试；Cirq 保留为可选交叉验证 |
| CPU 默认 | GPU 优先 | 网络与数据规模小，CPU 足够；降低复现门槛（无 CUDA 也能跑） |

---

## 14. 正确性验证矩阵（设计 → 测试）

| 不变量/设计点 | 测试文件 | 断言内容 |
| --- | --- | --- |
| 门矩阵幺正、模相位匹配 | `tests/test_gates.py` | 门库矩阵幺正；`match_single_qubit_gate` 对已知乘积命中/不命中 |
| 电路构造与序列化 | `tests/test_circuit.py` | 非法层报错、空层丢弃、cost/指标、`to_dict/from_dict` 往返 |
| 模拟器正确性 | `tests/test_simulator.py` | 已知门作用、任意电路幺正、与 cirq 一致（可选） |
| 随机电路与数据集 | `tests/test_generator.py` | 深度/合法性/可复现/JSON 往返 |
| **规则等价且降代价** | `tests/test_rules.py` | 随机 200 电路 × 4 规则：可适用 ⇒ cost 严格下降 + 等价 + 合法；不可适用 ⇒ `is_applicable=False` |
| 动作编码 | `tests/test_actions.py` | 编码/解码往返、与 `[R,Q,M]` 展平一致 |
| 观测编码 | `tests/test_observation.py` | 通道语义、padding、形状 |
| **掩码一致性** | `tests/test_masking.py` | 与 `applicable_actions` 一致；非法概率为 0；熵无 nan |
| 环境语义 | `tests/test_environment.py` | reset/step/奖励公式/done 分类/info 字段/非法动作抛错 |
| 网络与掩码采样 | `tests/test_agent.py` | 形状、非法动作概率 0、deterministic、梯度可反传 |
| **GAE/PPO 正确性** | `tests/test_ppo.py` | GAE 手算参考一致；ratio 初值 1；更新后参数变化；指标键齐全；KL 早停 |
| 采样器 | `tests/test_rollout.py` | 形状、done 处理、next_values bootstrap、mask 保存 |
| 训练闭环 | `tests/test_trainer.py` | 极小配置 2 轮跑通；产物文件齐全；可恢复 checkpoint |
| 评估与基线 | `tests/test_evaluate.py` | 等价率 1.0；指标合理；Markdown/JSON 输出 |
| 脚本 CLI | `tests/test_scripts.py` | 所有脚本 `--help` 可运行 |

---

## 15. 局限与可扩展方向

1. **规则不完备**：未覆盖参数化旋转门合并、三比特门、测量/重置、任意酉矩阵合成；
2. **策略局部视野**：3 层 3×3 卷积感受野 7×7，对跨 7 个时隙以上的长程依赖不敏感；
   可扩展为空洞卷积/全局池化/GNN；
3. **固定规模**：训练分布固定 5 比特、深度 12；跨规模泛化需要课程学习或结构适配；
4. **稠密随机电路上调度类规则触发少**：`move_*` 要求独占时隙，深度压缩率受限；
   可引入“生成更稀疏/更长电路”的训练分布或增加“门重排/交换”类规则；
5. **未建模真实硬件**：拓扑/噪声/保真度；可扩展为带耦合图约束的动作掩码与噪声感知奖励；
6. **未与工业转译器对比**：后续可加入 Qiskit/TKet 优化 pass 作为基线。

这些局限不会破坏 I1–I3，但决定了本设计是“一个正确、可复现的 RL 优化框架/毕业设计原型”，
而不是“超越工业转译器的 SOTA 系统”。这一点在答辩中会如实说明。
