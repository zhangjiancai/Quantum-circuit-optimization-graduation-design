"""实验可视化：训练曲线、电路图、方法对比与规则使用占比。

设计约束
--------
- 强制使用 ``Agg`` 后端，**不弹窗**，服务器 / 无显示器环境可直接出图；
- 中文字体（微软雅黑 / 黑体等）缺失时自动回退为英文标签，绝不因缺字体而崩溃；
- 电路图为**自绘**（横向比特线 + 门方框），不依赖 cirq，也不依赖任何门库绘图支持；
- 所有函数只接收文件路径，方便脚本与测试在临时目录中调用。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 必须在导入 pyplot 之前指定无界面后端
import matplotlib.pyplot as plt  # noqa: E402 - 后端设置之后再导入
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from qco.circuit import Circuit  # noqa: E402
from qco.gates import GATES  # noqa: E402
from qco.rules import RULES  # noqa: E402
from qco.utils import ensure_dir  # noqa: E402

#: 候选中文字体，按优先级排列（Windows 常见字体在列首）。
CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
)

#: 中文字体检测结果缓存：``None`` 表示尚未检测。
_CHINESE_FONT_READY: bool | None = None


def setup_matplotlib_chinese_font() -> bool:
    """尝试为 matplotlib 配置中文字体。

    依次检查 :data:`CHINESE_FONT_CANDIDATES` 中的字体是否已安装；命中时设置
    ``rcParams["font.sans-serif"]`` 并关闭负号乱码，返回 ``True``。
    若均未安装则返回 ``False``，且**不修改**全局字体配置——此时所有作图函数
    自动使用英文标签（见 :func:`_label`）。

    Returns
    -------
    bool
        是否成功启用中文字体。
    """
    global _CHINESE_FONT_READY
    if _CHINESE_FONT_READY is not None:
        return _CHINESE_FONT_READY
    try:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in CHINESE_FONT_CANDIDATES:
            if candidate in installed:
                matplotlib.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
                matplotlib.rcParams["font.family"] = "sans-serif"
                matplotlib.rcParams["axes.unicode_minus"] = False
                _CHINESE_FONT_READY = True
                return True
    except Exception:  # noqa: BLE001 - 字体子系统异常不应影响出图
        pass
    _CHINESE_FONT_READY = False
    return False


def _label(chinese: str, english: str) -> str:
    """按中文字体是否可用来选择标签文本。"""
    return chinese if setup_matplotlib_chinese_font() else english


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    """读取 CSV 为字典列表；文件不存在时抛 ``FileNotFoundError``。"""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any, default: float = float("nan")) -> float:
    """把 CSV 单元格安全转换为浮点数。"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _save(fig: Any, out_path: str | Path) -> None:
    """保存图形并释放画布，避免长时间训练中累积 figure。"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _empty_panel(axis: Any, message: str) -> None:
    """在空白面板上居中显示提示文字。"""
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def plot_training_curves(
    train_csv: str | Path,
    eval_csv: str | Path,
    out_path: str | Path,
) -> None:
    """绘制 2×2 训练曲线图。

    四个子图依次为：训练回报、评估集代价压缩率、门数压缩率、深度压缩率。
    评估曲线优先使用 ``method == "agent"`` 的行；没有 agent 行时退化为
    “每次评估全部方法各画一条线”。

    Parameters
    ----------
    train_csv
        ``train_log.csv`` 路径（列见 :class:`qco.trainer.Trainer`）。
    eval_csv
        ``eval_log.csv`` 路径。
    out_path
        输出 PNG 路径；父目录不存在时自动创建。
    """
    train_rows = _read_csv(train_csv)
    eval_rows = _read_csv(eval_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    figure_rows = [
        (
            axes[0, 0],
            "train",
            "mean_return",
            _label("训练回报（每次迭代）", "Training return (per iteration)"),
            _label("平均回报", "Mean return"),
        ),
        (
            axes[0, 1],
            "eval",
            "cost_reduction_pct",
            _label("评估集代价压缩率", "Eval cost reduction"),
            _label("代价压缩率 (%)", "Cost reduction (%)"),
        ),
        (
            axes[1, 0],
            "eval",
            "gate_reduction_pct",
            _label("评估集门数压缩率", "Eval gate reduction"),
            _label("门数压缩率 (%)", "Gate reduction (%)"),
        ),
        (
            axes[1, 1],
            "eval",
            "depth_reduction_pct",
            _label("评估集深度压缩率", "Eval depth reduction"),
            _label("深度压缩率 (%)", "Depth reduction (%)"),
        ),
    ]

    for axis, source, column, title, ylabel in figure_rows:
        rows = train_rows if source == "train" else eval_rows
        axis.set_title(title)
        axis.set_xlabel(_label("迭代轮次", "Iteration"))
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if not rows:
            _empty_panel(axis, _label("暂无数据", "No data"))
            continue
        if source == "eval":
            # 评估日志中优先只画 agent；没有 agent 行时按方法分组。
            agent_rows = [row for row in rows if row.get("method") == "agent"]
            if agent_rows:
                series = [
                    (
                        _label("PPO 智能体", "PPO agent"),
                        [_to_float(row.get("iteration")) for row in agent_rows],
                        [_to_float(row.get(column)) for row in agent_rows],
                    )
                ]
            else:
                methods: list[str] = []
                for row in rows:
                    method = str(row.get("method", ""))
                    if method not in methods:
                        methods.append(method)
                series = [
                    (
                        method,
                        [_to_float(row.get("iteration")) for row in rows if row.get("method") == method],
                        [_to_float(row.get(column)) for row in rows if row.get("method") == method],
                    )
                    for method in methods
                ]
        else:
            iterations = [_to_float(row.get("iteration")) for row in rows]
            values = [_to_float(row.get(column)) for row in rows]
            series = [(_label("训练", "Train"), iterations, values)]

        for name, xs, ys in series:
            valid = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
            if not valid:
                continue
            clean_x, clean_y = zip(*valid)
            axis.plot(clean_x, clean_y, marker="o", markersize=3, label=name)
        if len(series) > 1:
            axis.legend()

    fig.suptitle(_label("量子电路优化训练过程", "Quantum circuit optimization training"))
    _save(fig, out_path)


def _draw_two_qubit_gate(
    axis: Any,
    moment: int,
    op_qubits: Sequence[int],
    n_qubits: int,
    label: str,
    color: str,
) -> None:
    """绘制跨两个比特的门：竖直连线 + 跨比特方框。"""
    y_first = n_qubits - 1 - int(op_qubits[0])
    y_second = n_qubits - 1 - int(op_qubits[1])
    y_low, y_high = sorted((y_first, y_second))
    axis.plot(
        [moment, moment],
        [y_low, y_high],
        color="black",
        linewidth=1.2,
        zorder=2,
        solid_capstyle="butt",
    )
    rect = Rectangle(
        (moment - 0.36, y_low + 0.21),
        0.72,
        max(0.0, (y_high - y_low) - 0.42),
        facecolor=color,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    axis.add_patch(rect)
    axis.text(
        moment,
        (y_low + y_high) / 2.0,
        label,
        ha="center",
        va="center",
        fontsize=6.5,
        rotation=90 if y_high - y_low >= 0.9 else 0,
        zorder=4,
    )
    # 方框两端补上到比特线的短竖线，保证视觉连通。
    axis.plot([moment, moment], [y_low + 0.21, y_low], color="black", linewidth=1.0, zorder=2)
    axis.plot([moment, moment], [y_high, y_high - 0.21], color="black", linewidth=1.0, zorder=2)


def plot_circuit(circuit: Circuit, out_path: str | Path, title: str = "") -> None:
    """自绘电路图：横向比特线 + 门方框，输出 PNG（不依赖 cirq）。

    Parameters
    ----------
    circuit
        待绘制电路。
    out_path
        输出 PNG 路径。
    title
        图标题；为空时自动显示“门数 / 深度”摘要。
    """
    n_qubits = circuit.n_qubits
    depth = circuit.depth
    width = max(6.0, 0.95 * depth + 2.6)
    height = max(2.6, 0.95 * n_qubits + 1.2)
    fig, axis = plt.subplots(figsize=(width, height))

    # 比特线
    for qubit in range(n_qubits):
        y = n_qubits - 1 - qubit
        axis.plot([-0.5, max(depth - 0.5, 0.5)], [y, y], color="black", linewidth=1.0, zorder=1)
        axis.text(
            -0.8,
            y,
            f"q{qubit}",
            ha="right",
            va="center",
            fontsize=max(7.0, min(11.0, 9.0 - 0.3 * max(0, n_qubits - 5))),
        )

    single_color = "#cfe4ff"
    two_color = "#ffe0b3"
    for moment, layer in enumerate(circuit.layers):
        for op in layer:
            gate_label = GATES[op.gate].label
            if op.num_qubits == 1:
                y = n_qubits - 1 - op.qubits[0]
                axis.add_patch(
                    Rectangle(
                        (moment - 0.34, y - 0.28),
                        0.68,
                        0.56,
                        facecolor=single_color,
                        edgecolor="black",
                        linewidth=1.0,
                        zorder=3,
                    )
                )
                axis.text(
                    moment,
                    y,
                    gate_label,
                    ha="center",
                    va="center",
                    fontsize=6.5 if len(gate_label) > 2 else 8.0,
                    zorder=4,
                )
            else:
                _draw_two_qubit_gate(axis, moment, op.qubits, n_qubits, gate_label, two_color)

    axis.set_xlim(-1.3, max(depth - 0.5, 0.5) + 0.7)
    axis.set_ylim(-0.9, n_qubits - 0.1)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    if title:
        axis.set_title(title)
    else:
        axis.set_title(
            _label(
                f"电路图（门数 = {circuit.num_ops}，深度 = {depth}）",
                f"Circuit (gates = {circuit.num_ops}, depth = {depth})",
            )
        )
    _save(fig, out_path)


def plot_method_comparison(results: Mapping[str, Any], out_path: str | Path) -> None:
    """绘制不同方法的门数 / 深度 / 代价压缩率分组柱状图。

    Parameters
    ----------
    results
        方法名 -> :class:`qco.evaluate.EvalResult` 的映射。
    out_path
        输出 PNG 路径。
    """
    fig, axis = plt.subplots(figsize=(max(6.5, 1.6 * max(len(results), 1) + 3.0), 4.8))
    methods = list(results)
    metrics = [
        ("gate_reduction_pct", _label("门数压缩率", "Gate reduction")),
        ("depth_reduction_pct", _label("深度压缩率", "Depth reduction")),
        ("cost_reduction_pct", _label("代价压缩率", "Cost reduction")),
    ]
    if not methods:
        _empty_panel(axis, _label("暂无评估数据", "No evaluation data"))
        axis.set_title(_label("方法对比", "Method comparison"))
        _save(fig, out_path)
        return

    x = np.arange(len(methods), dtype=np.float64)
    width = 0.8 / len(metrics)
    for index, (field, label) in enumerate(metrics):
        values = [float(getattr(results[method], field, 0.0)) for method in methods]
        positions = x - 0.4 + width * (index + 0.5)
        bars = axis.bar(positions, values, width=width, label=label)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    axis.set_xticks(x)
    axis.set_xticklabels(methods)
    axis.set_ylabel(_label("压缩率 (%)", "Reduction (%)"))
    axis.set_title(_label("各方法优化效果对比", "Optimization comparison across methods"))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    _save(fig, out_path)


def plot_rule_usage(results: Mapping[str, Any], out_path: str | Path) -> None:
    """绘制各方法的规则使用占比堆叠柱状图。

    对每种方法，把 4 条规则的平均使用次数归一化为占比；没有使用任何规则的
    方法（如 noop）显示为空柱。

    Parameters
    ----------
    results
        方法名 -> :class:`qco.evaluate.EvalResult` 的映射。
    out_path
        输出 PNG 路径。
    """
    fig, axis = plt.subplots(figsize=(max(6.5, 1.6 * max(len(results), 1) + 3.0), 4.8))
    methods = list(results)
    rule_names = [rule.name for rule in RULES]
    if not methods:
        _empty_panel(axis, _label("暂无评估数据", "No evaluation data"))
        axis.set_title(_label("规则使用占比", "Rule usage"))
        _save(fig, out_path)
        return

    totals = []
    shares: list[list[float]] = [[] for _ in rule_names]
    for method in methods:
        counts = getattr(results[method], "mean_rule_counts", {}) or {}
        values = [float(counts.get(name, 0.0)) for name in rule_names]
        total = float(sum(values))
        totals.append(total)
        for index, value in enumerate(values):
            shares[index].append(value / total if total > 1e-12 else 0.0)

    x = np.arange(len(methods), dtype=np.float64)
    bottom = np.zeros(len(methods), dtype=np.float64)
    colors = matplotlib.colormaps["tab10"].colors
    for index, name in enumerate(rule_names):
        values = np.asarray(shares[index], dtype=np.float64)
        axis.bar(x, values, bottom=bottom, label=name, color=colors[index % len(colors)])
        bottom += values

    axis.set_xticks(x)
    axis.set_xticklabels(methods)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel(_label("规则使用占比", "Rule usage share"))
    axis.set_title(_label("各方法的规则使用占比", "Rule usage share by method"))
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="upper right", fontsize=8)
    _save(fig, out_path)
