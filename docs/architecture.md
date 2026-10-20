# 技术架构与接口契约（团队内部实现规范）

> 本文件是重写版毕业设计的**唯一接口契约**。所有模块必须严格按本文件的类名、函数签名、
> 张量形状、动作编码与数据约定实现，禁止自行改动跨模块接口。
> 项目根目录：`C:\Users\zjc20\quantum-circuit-optimization`
> Python 解释器：`C:\Users\zjc20\quantum-circuit-optimization\.venv\Scripts\python.exe`

---

## 0. 项目目标

把原仓库（PPO + 量子电路，代码不可运行）重写为**真正能优化量子电路**的完整工程：

- 领域：基于强化学习的量子电路**等价重写优化**（门合并、相邻消去、时隙压缩）。
- 状态：电路的门-比特-时隙张量 `[C, Q, M]`；
- 动作：`(规则 r, 量子比特 q, 时隙 m)` 三元组，共 `R*Q*M` 个，动作掩码屏蔽不可用动作；
- 奖励：势函数塑形 `r_t = (C_before - C_after) / C_initial`，其中
  `C = 门数量 + λ_depth × 电路深度`；
- 算法：CNN Actor-Critic + PPO（GAE、裁剪目标、熵正则、小批量多轮更新）；
- 保障：所有重写规则保持电路幺正矩阵**在全局相位意义下不变**（用状态向量模拟器验证）。

重写规则固定为 4 个（顺序即动作空间中的规则下标，严禁调换）：

| 下标 | 类名 | `name` | 含义 |
| --- | --- | --- | --- |
| 0 | `MergeSingleQubitRule` | `merge_single_qubit` | 同一量子比特上相邻两个单比特门相乘，若乘积（模全局相位）等于门库中某个门/单位阵，则合并为一个门，单位阵则删除 |
| 1 | `CancelTwoQubitRule` | `cancel_two_qubit` | 同一有序比特对上相邻的两个相同自逆二比特门（CNOT/CZ/SWAP）相互抵消 |
| 2 | `MoveEarlierRule` | `move_earlier` | 某门独占一个时隙时，把它前移到不违反各比特顺序的最早空闲时隙（深度 -1） |
| 3 | `MoveLaterRule` | `move_later` | 同上，向后移动到最晚空闲时隙（深度 -1） |

---

## 1. 门库约定（`qco/gates.py`）

门的编号顺序（也就是状态张量的通道顺序）固定为：

```
0 X, 1 Y, 2 Z, 3 H, 4 S, 5 Sdag, 6 T, 7 Tdag, 8 CNOT, 9 CZ, 10 SWAP
```

- 单比特门：`X, Y, Z, H, S, Sdag, T, Tdag`
  - `Sdag = diag(1, -i)`, `Tdag = diag(1, exp(-iπ/4))`
- 二比特门：`CNOT`（qubits 顺序为 `(control, target)`）、`CZ`、`SWAP`；
  二比特门矩阵按基矢顺序 `|00>, |01>, |10>, |11>`，比特 0 为高位。
- 单位阵以特殊名字 `"I"` 表示，**不是**可放置的门，只作为合并结果使用。
- `matrix` 为 `numpy.complex128`。

```python
@dataclass(frozen=True)
class GateSpec:
    name: str
    num_qubits: int
    matrix: np.ndarray            # 2**num_qubits 方阵
    label: str                    # 展示用，如 "X"、"CNOT"
    cirq_factory: Callable[[], Any] | None   # 延迟构造 cirq 门，可选依赖

GATE_NAMES: tuple[str, ...]              # 固定顺序，见上
SINGLE_QUBIT_GATE_NAMES: tuple[str, ...]
TWO_QUBIT_GATE_NAMES: tuple[str, ...]
SELF_INVERSE_TWO_QUBIT: tuple[str, ...]  # ("CNOT", "CZ", "SWAP")
GATES: dict[str, GateSpec]

GATE_CHANNEL_INDEX: dict[str, int]       # 门名 -> 通道下标（0..10）
CONTROL_CHANNEL: int = 11
TARGET_CHANNEL: int = 12
IDLE_CHANNEL: int = 13
N_CHANNELS: int = 14                     # = len(GATE_NAMES) + 3

def gate_matrix(name: str) -> np.ndarray
def gate_matrix_on(gate: str, qubits: Sequence[int], n_qubits: int) -> np.ndarray
    """返回嵌入到 n_qubits 系统中的 2**n × 2**n 矩阵（其他比特为单位阵）。"""
def match_single_qubit_gate(matrix: np.ndarray, atol: float = 1e-8) -> str | None
    """把 2×2 幺正矩阵在“模全局相位”意义下匹配到单比特门库；
    命中返回门名（"X"/.../"Tdag"），等于单位阵返回 "I"，否则 None。"""
def is_unitary(matrix: np.ndarray, atol: float = 1e-8) -> bool
def global_phase_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool
```

---

## 2. 电路表示（`qco/circuit.py`）

**不可变**数据结构；所有重写规则返回新 `Circuit`，禁止原地修改。

```python
@dataclass(frozen=True)
class Op:
    gate: str                    # GATE_NAMES 中的门名
    qubits: tuple[int, ...]      # CNOT 顺序为 (control, target)；其余有序

class Circuit:
    def __init__(self, n_qubits: int, layers: Iterable[Iterable[Op]] = ())
        # 构造时校验：比特下标合法、同一层内同一比特最多一个门、不允许多比特门重复比特；
        # 自动丢弃空层；layers 不可变（tuple of tuple）。
    n_qubits: int
    layers: tuple[tuple[Op, ...], ...]
    depth: int                                # 层数（= 时隙数）
    num_ops: int
    num_single_qubit_ops: int
    num_two_qubit_ops: int
    def cost(self, lambda_depth: float = 0.5) -> float
    def layer(self, index: int) -> tuple[Op, ...]
    def layer_qubits(self, index: int) -> frozenset[int]
    def op_on(self, qubit: int, layer: int) -> Op | None
    def next_op_on(self, qubit: int, layer: int) -> tuple[int, Op] | None   # 严格在 layer 之后的下一个触及该比特的门
    def prev_op_on(self, qubit: int, layer: int) -> tuple[int, Op] | None
    def qubit_ops(self, qubit: int) -> list[tuple[int, Op]]
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> "Circuit"
    def to_cirq(self) -> Any                  # 可选依赖 cirq；未安装抛 RuntimeError
    @classmethod
    def from_cirq(cls, circuit: Any) -> "Circuit"
    @property
    def unitary(self) -> np.ndarray           # 2**n × 2**n，延迟计算并缓存（调用 simulator）
    def equivalent(self, other: "Circuit", atol: float = 1e-8) -> bool
    def to_text(self) -> str                  # 可读文本：有 cirq 用 cirq 图，否则自绘 ASCII
    def __eq__(self, other) -> bool           # 仅比较 n_qubits 与 layers
    def __hash__(self) -> int
```

序列化格式（`Circuit.to_dict`，供数据集 JSON 使用）：

```python
{"n_qubits": 5, "layers": [[["X", [0]], ["CNOT", [1, 2]]], [["H", [3]]]]}
```

---

## 3. 模拟器（`qco/simulator.py`）

纯 numpy 实现，**不依赖 cirq**；比特 0 为最高位（big-endian）。

```python
def apply_gate_to_state(state: np.ndarray, gate: str, qubits: Sequence[int], n_qubits: int) -> np.ndarray
def circuit_state_vector(circuit: Circuit) -> np.ndarray      # 初态 |0...0>
def circuit_unitary(circuit: Circuit) -> np.ndarray
def equivalent_up_to_global_phase(u: np.ndarray, v: np.ndarray, atol: float = 1e-8) -> bool
def circuits_equivalent(a: Circuit, b: Circuit, atol: float = 1e-8) -> bool
```

测试必须全部通过：

- `H`、`X`、`CNOT`、`SWAP`、`T` 的已知作用结果；
- 任意 `Circuit` 的 `unitary` 为幺正阵；
- 若安装了 cirq，则 `cirq.unitary(circuit.to_cirq())` 与 `circuit_unitary(circuit)` 在全局相位意义下相等（用 `cirq.testing.assert_allclose_up_to_global_phase` 或自写比较）。

---

## 4. 观测编码（`qco/observation.py`）

```python
def encode_circuit(circuit: Circuit, n_moments: int) -> np.ndarray
    """返回形状 [N_CHANNELS, n_qubits, n_moments] 的 float32 数组。
    对每个时隙 m（0 <= m < n_moments）：
    - 单比特门 op：channel[GATE_CHANNEL_INDEX[op.gate], q, m] = 1；
    - 二比特门 op：两个比特都置门通道 1；且 channel[CONTROL_CHANNEL, q0, m] = 1,
      channel[TARGET_CHANNEL, q1, m] = 1（q0, q1 按 op.qubits 顺序）；
    - 该时隙空闲的比特：channel[IDLE_CHANNEL, q, m] = 1；
    - 时隙超出电路实际深度时：所有比特在 IDLE_CHANNEL 置 1。
    circuit.depth > n_moments 时抛 ValueError。"""
def observation_shape(n_qubits: int, n_moments: int) -> tuple[int, int, int]   # (N_CHANNELS, Q, M)
```

---

## 5. 随机电路与数据集（`qco/generator.py`）

```python
def random_circuit(n_qubits: int, depth: int, rng: np.random.Generator,
                   p_two: float = 0.15) -> Circuit
    """逐层生成：随机顺序遍历比特；未占用比特以 p_two 概率与相邻比特 (q, q+1) 放二比特门，
    否则放随机单比特门；每层保证至少一个门；结果深度恰为 depth（丢弃空层后补足）。"""
def random_circuits(count: int, n_qubits: int, depth: int, seed: int = 0,
                    p_two: float = 0.15) -> list[Circuit]
def save_circuits(circuits: Sequence[Circuit], path: str | Path) -> None
    # JSON: {"meta": {"count": n, "n_qubits": q, "depth": d}, "circuits": [to_dict, ...]}
def load_circuits(path: str | Path) -> list[Circuit]
```

---

## 6. 动作编码（`qco/actions.py`）

固定布局 `index = (rule * n_qubits + qubit) * n_moments + moment`。

```python
def action_space_size(n_rules: int, n_qubits: int, n_moments: int) -> int
def encode_action(rule: int, qubit: int, moment: int, n_qubits: int, n_moments: int) -> int
def decode_action(index: int, n_qubits: int, n_moments: int) -> tuple[int, int, int]
def encode_actions(rule, qubit, moment, n_qubits, n_moments) -> np.ndarray   # 向量化
```

---

## 7. 重写规则（`qco/rules.py`）

```python
class RewriteRule(abc.ABC):
    name: str
    description: str
    rule_index: int                 # 与 RULES 顺序一致
    def is_applicable(self, circuit: Circuit, qubit: int, moment: int) -> bool
    def apply(self, circuit: Circuit, qubit: int, moment: int) -> Circuit   # 前置条件：is_applicable 为 True

RULES: tuple[RewriteRule, ...]      # 顺序见第 0 节，长度 = 4
RULES_BY_NAME: dict[str, RewriteRule]
def applicable_actions(circuit: Circuit) -> list[tuple[int, int, int]]   # (rule_index, qubit, moment)
```

各规则语义（必须严格实现）：

1. `MergeSingleQubitRule`
   - `op1 = circuit.op_on(q, m)`；要求 `op1` 存在且为单比特门；
   - 找 `op1` 之后（时隙 > m）该比特上的下一个门 `op2`；要求 `op2` 也是单比特门；
   - 令 `U = matrix(op2) @ matrix(op1)`（先 op1 后 op2），用
     `match_single_qubit_gate(U)` 匹配门库；
   - 命中 `"I"`：删除两个门；否则在时隙 m 处放合并后的门。返回新电路。
2. `CancelTwoQubitRule`
   - `op1 = circuit.op_on(q, m)`；要求 `op1` 是二比特门，且 `op1.gate in SELF_INVERSE_TWO_QUBIT`；
   - `op1` 的两个比特 `a, b`：`next_op_on(a, m)` 与 `next_op_on(b, m)` 必须是**同一层、同一个门**
     （即 `op2 == op1`），删除这两层中的这两个门。
3. `MoveEarlierRule`
   - 要求 `len(layer(m)) == 1`（该门独占一层，保证深度会减少）；
   - 向 m-1 方向扫描，跳过所有与 `op.qubits` 不相交的层，得到可放置的最早位置 `p < m`；
   - 删除原层，把门放进层 p（其余层顺序不变，空层自动丢弃）。
4. `MoveLaterRule`
   - 要求 `len(layer(m)) == 1`；
   - 向 m+1 方向扫描，跳过与 `op.qubits` 不相交的层，得到可放置的最晚位置 `p > m`；
   - 删除原层，把门放进层 p。

**不变量**：任一 `is_applicable` 为真的动作，`apply` 后：
`new_cost < old_cost`（`cost = num_ops + λ_depth * depth`），且 `circuits_equivalent(old, new)`。

---

## 8. 动作掩码（`qco/masking.py`）

```python
def build_action_mask(circuit: Circuit, n_moments: int) -> np.ndarray
    """返回形状 [len(RULES), circuit.n_qubits, n_moments] 的 bool 数组：
    mask[r, q, m] = RULES[r].is_applicable(circuit, q, m) and m < circuit.depth。"""
def flatten_action_mask(mask: np.ndarray) -> np.ndarray                 # [R*Q*M]
def unflatten_action_mask(flat: np.ndarray, n_rules: int, n_qubits: int, n_moments: int) -> np.ndarray
def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
    """logits [..., R, Q, M] 或 [..., R*Q*M]，mask 可广播 bool；非法动作概率严格为 0。
    归一化沿最后一维进行；PPO/采样路径一律先展平为 [B, R*Q*M] 再调用。"""
def masked_entropy(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
```

`n_moments` 取 `EnvConfig.n_moments`；时隙下标 >= 电路深度时掩码恒为 False。

---

## 9. 环境（`qco/environment.py`）

```python
@dataclass(frozen=True)
class EnvConfig:
    n_qubits: int = 5
    n_moments: int = 15
    max_steps: int = 48
    lambda_depth: float = 0.5
    initial_depth: int = 12        # reset() 未给电路时随机生成的深度
    p_two: float = 0.15
    seed: int | None = 0
    # 校验：n_qubits >= 2、initial_depth <= n_moments、0 <= p_two <= 1、
    #       lambda_depth > 0（move_* 规则只改深度，λ=0 会破坏严格进步不变量）

class CircuitOptimizationEnv:
    def __init__(self, config: EnvConfig, circuit: Circuit | None = None,
                 rng: np.random.Generator | None = None)
    @property
    def n_actions(self) -> int                     # R*Q*M
    @property
    def circuit(self) -> Circuit
    @property
    def observation(self) -> np.ndarray            # [C, Q, M] float32
    @property
    def action_mask(self) -> np.ndarray            # [R, Q, M] bool
    @property
    def valid_actions(self) -> np.ndarray          # 一维 int64 动作下标
    @property
    def cost(self) -> float
    @property
    def initial_cost(self) -> float
    @property
    def steps(self) -> int
    def reset(self, circuit: Circuit | None = None) -> np.ndarray
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]
        """返回 (obs, reward, done, info)。
        - 非法动作：抛 ValueError（调用方必须先掩码）；
        - reward = (cost_before - cost_after) / initial_cost；
        - done = (无可用动作) or (steps >= max_steps)；
        - info: {rule, rule_name, qubit, moment, cost, initial_cost, gate_count, depth,
                 valid_actions, terminated, truncated, rule_counts(累计 {name: n})}。
        """
    def render(self) -> str
```

---

## 10. 策略网络（`qco/agent.py`）

```python
class CircuitPolicyNet(nn.Module):
    def __init__(self, in_channels: int = N_CHANNELS, n_rules: int = len(RULES),
                 n_qubits: int = 5, n_moments: int = 15, hidden_channels: int = 64)
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
        # observations: [B, C, Q, M] float32
        # 返回 logits [B, R, Q, M]（未 softmax 的原始打分）, value [B]
    def policy_logits(self, observations: torch.Tensor) -> torch.Tensor   # [B, R, Q, M]
    def act(self, observations: torch.Tensor, masks: torch.Tensor,
            deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        # 返回 (action [B] int64, log_prob [B], value [B], entropy [B])
```

网络结构（可微调但必须保持输入/输出形状）：

- 共享卷积主干：`Conv2d(C, H, 3, padding=1) + ELU` × 3（H = hidden_channels），
  使用 `GroupNorm` 或 `ELU`（**禁止 BatchNorm**：PPO 单样本/小批量训练不稳定）；
- 策略头：`Conv2d(H, R, 1)` → `[B, R, Q, M]`；
- 价值头：`Conv2d(H, 32, 1) → ELU → Flatten → Linear(32*Q*M, 128) → ELU → Linear(128, 1)` → `[B]`。
- `act` 内部：masked log_softmax → Categorical 采样/argmax → 返回对应 log_prob、value、entropy（
  熵按掩码归一化：`-(p*logp).sum(valid)/1` 直接用 log_probs 加权求和即可）。

---

## 11. Rollout（`qco/rollout.py`）

```python
@dataclass
class RolloutBatch:
    observations: torch.Tensor   # [num_envs, horizon, C, Q, M] float32
    actions: torch.Tensor        # [num_envs, horizon] int64
    log_probs: torch.Tensor      # [num_envs, horizon] float32
    values: torch.Tensor         # [num_envs, horizon] float32
    rewards: torch.Tensor        # [num_envs, horizon] float32
    dones: torch.Tensor          # [num_envs, horizon] float32（1.0 = 回合终止/截断）
    masks: torch.Tensor          # [num_envs, horizon, R, Q, M] bool
    next_values: torch.Tensor    # [num_envs] float32，最后一步之后的 bootstrap 价值
    def flatten(self) -> ...     # 提供展开视图（PPO 用）
@dataclass
class EpisodeStats:
    return_: float
    initial_cost: float
    final_cost: float
    initial_gate_count: int
    final_gate_count: int
    initial_depth: int
    final_depth: int
    steps: int
    truncated: bool
    rule_counts: dict[str, int]
class RolloutCollector:
    def __init__(self, policy: CircuitPolicyNet, envs: Sequence[CircuitOptimizationEnv],
                 device: torch.device | str = "cpu")
    def collect(self, horizon: int, circuit_sampler: Callable[[], Circuit]) -> tuple[RolloutBatch, list[EpisodeStats]]
        """并行采样：每个 env 连续走 horizon 步；若某 env 在本步 done（无可用动作或达到 max_steps），
        该步 dones=1，随后用 circuit_sampler() 取新电路 reset。next_values[env] 取“最后一步之后
        观测”的 bootstrap 价值（新回合的初始观测），GAE 用 (1-done) 截断，因此不受 auto-reset 影响。"""
```

---

## 12. PPO（`qco/ppo.py`）

```python
@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    target_kl: float = 0.02
    normalize_advantage: bool = True

class PPO:
    def __init__(self, policy: CircuitPolicyNet, config: PPOConfig, device: torch.device | str = "cpu")
    def update(self, batch: RolloutBatch) -> dict[str, float]
        """按 [num_envs, horizon] 逐环境计算 GAE、returns，再展平为样本；
        小批量多轮更新；返回指标：
        policy_loss, value_loss, entropy, approx_kl, clip_fraction, total_loss,
        explained_variance, learning_rate, grad_norm"""
    @staticmethod
    def compute_gae(rewards, values, dones, next_values, gamma, gae_lambda)
        -> tuple[torch.Tensor, torch.Tensor]      # advantages, returns，形状同输入
```

**正确性硬性要求**（会被独立审计）：

- GAE 反向递推须包含最后一步；`delta_t = r_t + γ * v_{t+1} * (1-d_t) - v_t`，
  `v_{t+1}` 在 t 为最后一个时取 `next_values[env]`；`A_t = delta_t + γλ(1-d_t)A_{t+1}`；
- `returns = advantages + values`；
- 策略损失用 `old_log_probs` 作为分母基准，ratio = exp(new - old)；
- 价值损失用 `returns`（detach），不使用裁剪价值除非明确实现；
- 熵项按有效动作计算；更新时重新前向必须使用**同一组动作掩码**；
- 不做双重 softmax（原代码的致命错误）：网络输出 logits，`masked_log_softmax` 只做一次。

---

## 13. 基线与评估（`qco/baselines.py`、`qco/evaluate.py`）

`baselines.py`：

```python
@dataclass
class OptimizationResult:
    method: str
    initial_circuit: Circuit
    final_circuit: Circuit
    initial_cost: float
    final_cost: float
    initial_gate_count: int
    final_gate_count: int
    initial_depth: int
    final_depth: int
    steps: int
    truncated: bool
    equivalent: bool
    rule_counts: dict[str, int]
    elapsed_ms: float
def optimize(env: CircuitOptimizationEnv, chooser: Callable[[CircuitOptimizationEnv], int],
             max_steps: int | None = None) -> OptimizationResult
def optimize_noop(circuit: Circuit, config: EnvConfig) -> OptimizationResult
def optimize_random(circuit: Circuit, config: EnvConfig, seed: int = 0) -> OptimizationResult
def optimize_greedy(circuit: Circuit, config: EnvConfig) -> OptimizationResult
def choose_random(env, rng) -> int
def choose_greedy(env) -> int      # 从 valid_actions 里选“一步后 cost 最小”的动作（并列取最小下标）
```

`evaluate.py`：

```python
@dataclass
class EvalResult:
    method: str
    n_circuits: int
    mean_initial_gate_count: float
    mean_final_gate_count: float
    mean_initial_depth: float
    mean_final_depth: float
    mean_initial_cost: float
    mean_final_cost: float
    gate_reduction_pct: float        # (init-final)/init * 100
    depth_reduction_pct: float
    cost_reduction_pct: float
    equivalence_rate: float          # 保持幺正等价的比例，应 == 1.0
    mean_steps: float
    truncation_rate: float
    mean_rule_counts: dict[str, float]
    mean_elapsed_ms: float
def evaluate_agent(policy, circuits, config: EnvConfig, device="cpu", deterministic=True) -> EvalResult
def evaluate_noop(circuits, config) -> EvalResult
def evaluate_random(circuits, config, seed=0) -> EvalResult
def evaluate_greedy(circuits, config) -> EvalResult
def format_comparison(results: dict[str, EvalResult]) -> str    # Markdown 表格
def save_eval_results(results: dict[str, EvalResult], path) -> None
```

评估时对每个电路检查 `circuits_equivalent(initial, final)`。

---

## 14. 训练器（`qco/trainer.py`）

```python
@dataclass(frozen=True)
class TrainConfig:
    env: EnvConfig = EnvConfig()
    ppo: PPOConfig = PPOConfig()
    iterations: int = 150
    steps_per_iter: int = 16          # horizon
    num_envs: int = 8
    hidden_channels: int = 64
    eval_every: int = 10
    save_every: int = 25
    eval_circuits: int = 100
    train_circuits: int = 200
    eval_seed: int = 10000
    out_dir: str = "results/default"
    device: str = "cpu"
class Trainer:
    def __init__(self, config: TrainConfig)
    def train(self) -> dict
        """产物（写入 config.out_dir）：
        - config.json    完整配置
        - train_log.csv  每个 iteration 一行：iteration, mean_return, mean_episode_len,
                         mean_initial_cost, mean_final_cost, gate_reduction_pct, depth_reduction_pct,
                         policy_loss, value_loss, entropy, approx_kl, explained_variance
        - eval_log.csv   每次评估一行：iteration, method, cost_reduction_pct, gate_reduction_pct,
                         depth_reduction_pct, equivalence_rate, mean_steps
        - best.pt / last.pt   模型 state_dict（含 config）
        - summary.json   最终评估（noop/random/greedy/agent）
        """
```

训练集：`random_circuits(train_circuits, Q, initial_depth, seed=env.seed)`；
评估集：若存在 `data/eval.json` 则加载，否则用 `eval_seed` 现生成 `eval_circuits` 个。
每 `eval_every` 轮在评估集上评估 agent（deterministic=True）并记录；`best.pt` 以评估集
`mean_final_cost` 最小为准。

---

## 15. 可视化与工具（`qco/visualizer.py`、`qco/utils.py`）

```python
# visualizer.py
def setup_matplotlib_chinese_font() -> bool        # 尝试 Microsoft YaHei / SimHei
def plot_training_curves(train_csv, eval_csv, out_path) -> None
    # 2×2 子图：回报、评估代价、门数压缩率、深度压缩率
def plot_circuit(circuit: Circuit, out_path, title: str = "") -> None
    # 自绘：横向比特线 + 门方框，输出 png（不依赖 cirq）
def plot_method_comparison(results: dict[str, EvalResult], out_path) -> None   # 分组柱状图
def plot_rule_usage(results: dict[str, EvalResult], out_path) -> None          # 各方法规则使用占比
# utils.py
def set_seed(seed: int) -> None            # random/numpy/torch 全量设种
def resolve_device(name: str = "cpu") -> torch.device
def ensure_dir(path) -> Path
def save_json(obj, path) -> None
def load_json(path) -> Any
def to_jsonable(obj) -> Any                # dataclass/numpy 递归转换
def config_to_dict(config) -> dict
```

---

## 16. 脚本（`scripts/`）与根目录入口

所有脚本使用 `argparse`，从项目根目录运行，默认路径相对项目根：

```
scripts/make_dataset.py  --out data/eval.json --count 100 --n-qubits 5 --depth 12 --seed 10000
scripts/train.py         --out results/default --iterations 150 --num-envs 8 --steps-per-iter 16
                         --seed 0 --device cpu --max-steps 48 --lambda-depth 0.5
scripts/evaluate.py      --checkpoint results/default/best.pt --dataset data/eval.json
                         --out results/default/eval --baselines random,greedy
scripts/make_figures.py  --run results/default --out results/figures
```

- `scripts/train.py` 打印中文进度日志（logging），训练结束打印 summary 表格；
- `scripts/evaluate.py` 输出 `evaluation.json` + Markdown 表格到 stdout；
- 根目录 `train.py` / `evaluate.py` 为薄封装，等价于 `scripts/train.py` / `scripts/evaluate.py`
  （便于对照原仓库文件名）；
- 所有脚本必须支持 `--help`，且在没有 GPU 的环境下可运行。

---

## 17. 测试要求（`tests/`）

所有测试用 `pytest`，禁止真实联网、禁止调用 cirq 之外的重依赖；CPU 上整套测试 < 120 秒。

必须覆盖：

1. `test_gates.py`：门矩阵幺正、模相位匹配、`gate_matrix_on` 与 Kronecker 结果一致；
2. `test_circuit.py`：构造校验/空层丢弃/指标/cost/序列化往返/to_cirq 往返（无 cirq 跳过）；
3. `test_simulator.py`：已知门作用、幺正性、随机电路与 cirq 计算一致；
4. `test_generator.py`：深度/合法性/可复现/数据集存取；
5. `test_rules.py`：**随机 200 个电路 × 4 规则**：可适用则应用后（a）cost 严格下降；
   （b）`circuits_equivalent` 为真；（c）电路合法；不可适用时 `is_applicable` 为假；
6. `test_masking.py`：掩码与 `applicable_actions` 一致、非法位置为 False、masked 概率为 0；
7. `test_environment.py`：reset/step/奖励公式/done/截断/info 字段/非法动作抛错；
8. `test_agent.py`：形状、掩码后非法动作概率 0、deterministic 选择、梯度可反传；
9. `test_ppo.py`：GAE 与逐项手算/参考循环一致、ratio 初值为 1、更新后参数变化、
   指标键齐全、KL 早停不报错；
10. `test_trainer.py`：极小配置 2 iterations 跑通、产物文件齐全、可恢复 checkpoint；
11. `test_evaluate.py`：三种基线指标合理（等价率 1.0、随机不优于 greedy 的断言放宽）；
12. `test_scripts.py`：脚本 `--help` 能跑（subprocess 或 import main 函数）。
13. 独立审计（T4）：对规则/GAE/掩码做对抗性测试并出具报告。

---

## 18. 代码风格与协作约定

- Python 3.10+ 类型注解；每个公开类/函数写**中文** docstring（本科毕业设计中文语境）；
- 模块顶部 `from __future__ import annotations`；
- 注释解释“为什么/约定”，不逐行翻译代码；
- 禁止 `print` 调试残留（脚本里的用户输出用 `logging` / `print` 仅限脚本主流程）；
- 禁止改动他人负责的文件；跨文件接口问题先改本文件（契约）并通知 Lead；
- 提交前跑：`.venv\Scripts\python.exe -m pytest tests -q` 与
  `.venv\Scripts\python.exe -m compileall qco scripts train.py evaluate.py`；
- 不使用 git 提交、不安装新依赖、不联网。
