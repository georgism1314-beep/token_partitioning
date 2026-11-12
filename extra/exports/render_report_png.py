"""把 Factor Decomp Report 渲成一张长图 PNG（不依赖 Cursor canvas 原生导出）。"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np

out = Path(__file__).resolve().parent / "olmo3-factor-decomp-report.png"
layers = ["L0", "L4", "L8", "L16", "L24", "L31"]
tok = np.array([0.931, 0.518, 0.277, 0.191, 0.132, 0.120])
d_prev = np.array([0.001, 0.146, 0.143, 0.103, 0.081, 0.076])
d_prev2 = np.array([0.0, 0.028, 0.047, 0.036, 0.029, 0.028])
resid = np.array([0.068, 0.292, 0.519, 0.660, 0.755, 0.768])
d_pos = np.array([-0.0021, 0.0126, 0.0118, 0.0094, 0.0076, 0.0078])
cc = np.array([0.037, 0.021, 0.020, 0.014, 0.011, 0.011])
pos = np.array([0.042, 0.026, 0.024, 0.020, 0.018, 0.018])

BG, CARD, FG, MUTED, STROKE = "#F7F7F5", "#FFFFFF", "#1A1A1A", "#5C5C5C", "#E5E5E2"
ACCENT, WARN, OK, GRAY = "#2563EB", "#B45309", "#15803D", "#6B7280"
INFO_BG, WARN_BG, NEUT_BG = "#EFF6FF", "#FFFBEB", "#F4F4F2"


def box(ax, title, body, facecolor, y0=0.98):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0, 0.02), 0.998, 0.96, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1, edgecolor=STROKE, facecolor=facecolor, transform=ax.transAxes))
    ax.text(0.02, 0.78, title, transform=ax.transAxes, fontsize=11, fontweight="700", color=FG, va="top")
    ax.text(0.02, 0.58, body, transform=ax.transAxes, fontsize=9.5, color=FG, va="top", wrap=True,
            linespacing=1.35)


def style_ax(ax, ymax=None):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(STROKE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=STROKE, linewidth=0.8)
    ax.set_axisbelow(True)
    if ymax is not None:
        ax.set_ylim(0, ymax)


plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
})

# 中文：从系统 NotoSansCJK.ttc 抽出简体（index 2），再注册给 matplotlib（直接 addfont TTC 会落到 JP）
from matplotlib import font_manager
from fontTools.ttLib import TTCollection
cjk_otf = Path(__file__).resolve().parent / "NotoSansCJKsc-Regular.otf"
if not cjk_otf.exists():
    TTCollection("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").fonts[2].save(cjk_otf)
font_manager.fontManager.addfont(str(cjk_otf))
cjk = font_manager.FontProperties(fname=str(cjk_otf)).get_name()
plt.rcParams["font.family"] = cjk
plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]

fig = plt.figure(figsize=(12.5, 28), dpi=160)
gs = GridSpec(18, 4, figure=fig, height_ratios=[0.7, 0.55, 1.05, 1.05, 1.05, 0.35, 0.95, 2.2, 1.35, 1.05,
                                               0.35, 0.95, 2.0, 1.05, 0.7, 1.4, 0.3, 0.25],
              hspace=0.55, wspace=0.25, left=0.06, right=0.97, top=0.985, bottom=0.015)

# 标题
ax = fig.add_subplot(gs[0, :])
ax.axis("off")
ax.text(0, 0.72, "因素分解进展 · tok / prev / trigram / POS", fontsize=20, fontweight="700", color=FG, va="top")
ax.text(0, 0.18,
        "相对上周：上周只看了 token-id η² 随层深下降；本周完成 prev/bigram 嵌套分解，并扩到 trigram 与 spaCy POS。"
        "读缓存激活（12 ckpt × 6 层），原始 silu(gate)·up，cross-fit η² 中位数。",
        fontsize=10, color=MUTED, va="top", wrap=True)

# 四个 stat
stats = [
    ("0.146", "step10k L4 · Δ(prev|tok)", ACCENT),
    ("0.047", "step10k L8 · Δ(prev2|bigram)", FG),
    ("0.013", "step10k L4 · Δ(POS|tok)", WARN),
    ("98–100%", "神经元先选 tok（贪心第一步）", OK),
]
for i, (v, lab, col) in enumerate(stats):
    ax = fig.add_subplot(gs[1, i])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1, edgecolor=STROKE, facecolor=CARD, transform=ax.transAxes))
    ax.text(0.08, 0.62, v, transform=ax.transAxes, fontsize=18, fontweight="700", color=col, va="center")
    ax.text(0.08, 0.22, lab, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="center")

# 上周三段
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
ax.text(0, 1.05, "回顾：上周汇报（token-id 随层下降）", fontsize=14, fontweight="650", color=FG, transform=ax.transAxes, va="top")
box(ax, "现象",
    "L0 FFN 几乎直接读词嵌入相关的 token 信号。训练开始后，层越深，由「当前 token」决定的方差比例越小、由上下文决定的比例越大"
    "——不等于「深层不需要 token 信息」。L31 与 L24 基本持平。step 1k 全层跳升，之后深层回落、5k+ 平台。",
    NEUT_BG)

ax = fig.add_subplot(gs[3, :])
box(ax, "解释",
    "Olmo-3 后置归一化：h_l = e_t + Σ (a_k + m_k) + a_l；attention/FFN 输入不做 norm。FFN 读全部写入块累加，"
    "token-id η² ≈ 各块 η² 的方差加权平均；注意力是唯一跨位置搬运信息的操作，浅层 token 信号被稀释。"
    "1k 跳变来自 L0 注意力变成 token 特异；基座 L0 神经元 η² 仍只有 ~0.48。",
    INFO_BG)

ax = fig.add_subplot(gs[4, :])
box(ax, "上周 todo → 本周已完成",
    "补「前一个 token」因素，做嵌套分解：先 token，再 (token, 前一个 token)，看深层「上下文」方差里有多少只是 bigram。"
    "本周已实现，并扩到 trigram；另加字符类与 spaCy POS。",
    WARN_BG)

# 分隔 + 本周一
ax = fig.add_subplot(gs[5, :])
ax.axis("off")
ax.plot([0, 1], [0.5, 0.5], color=STROKE, lw=1, transform=ax.transAxes, clip_on=False)
ax.text(0, 0.0, "本周一：嵌套 n-gram 分解   [step 10k]", fontsize=14, fontweight="650", color=FG, transform=ax.transAxes, va="bottom")

ax = fig.add_subplot(gs[6, :])
box(ax, "现象",
    "η²(tok)+Δ(prev|tok)+Δ(prev2|bigram)+剩余=1。训练后 L0 几乎全是 tok；L4 起 prev 增量明显，L4–L8 最高（~0.14–0.15）。"
    "trigram 增量更小且几乎不涨（峰值~0.05）。深层剩余仍大（L31 剩余 0.77）。贪心每轮重算：第一步几乎总选 tok（98–100%）。",
    NEUT_BG)

ax = fig.add_subplot(gs[7, :])
style_ax(ax, 1.0)
x = np.arange(len(layers))
ax.plot(x, tok, "o-", color=FG, lw=2, ms=5, label="η²(tok)")
ax.plot(x, d_prev, "o-", color=ACCENT, lw=2, ms=5, label="Δ(prev|tok)")
ax.plot(x, d_prev2, "o-", color=WARN, lw=2, ms=5, label="Δ(prev2|bigram)")
ax.plot(x, resid, "o--", color=GRAY, lw=2, ms=5, label="剩余 1−η²(trigram)")
ax.set_xticks(x); ax.set_xticklabels(layers)
ax.set_ylabel("cross-fit η² 中位数")
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
ax.set_title("嵌套链（step 10k）", fontsize=11, color=MUTED, loc="left", pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

# 表格
ax = fig.add_subplot(gs[8, :])
ax.axis("off")
tbl = ax.table(
    cellText=[
        ["L0", "93.1%", "0.1%", "0.0%", "6.8%", "−0.2%"],
        ["L4", "51.8%", "14.6%", "2.8%", "29.2%", "1.3%"],
        ["L8", "27.7%", "14.3%", "4.7%", "51.9%", "1.2%"],
        ["L16", "19.1%", "10.3%", "3.6%", "66.0%", "0.9%"],
        ["L24", "13.2%", "8.1%", "2.9%", "75.5%", "0.8%"],
        ["L31", "12.0%", "7.6%", "2.8%", "76.8%", "0.8%"],
    ],
    colLabels=["层", "η²(tok)", "Δ(prev|tok)", "Δ(prev2|bigram)", "剩余", "对比 Δ(POS|tok)"],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.45)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(STROKE)
    if r == 0:
        cell.set_facecolor("#F0F0ED")
        cell.set_text_props(color=MUTED, fontweight="600")
    else:
        cell.set_facecolor(CARD)

ax = fig.add_subplot(gs[9, :])
box(ax, "解释",
    "分组：tok ⊂ bigram ⊂ trigram 严格嵌套；prev 是 bigram 按前一个 token 粗化。Δ(prev|tok)=η²(bigram)−η²(tok) "
    "量的是「给定当前 token 后，前一个 token 还多解释多少」。trigram 增量不涨：条件信息量小、覆盖约一半、更长程进剩余。"
    "神经元读到的上下文远比「前两个 token」细。",
    INFO_BG)

# 本周二
ax = fig.add_subplot(gs[10, :])
ax.axis("off")
ax.plot([0, 1], [0.5, 0.5], color=STROKE, lw=1, transform=ax.transAxes, clip_on=False)
ax.text(0, 0.0, "本周二：字符类 + 上下文 POS", fontsize=14, fontweight="650", color=FG, transform=ax.transAxes, va="bottom")

ax = fig.add_subplot(gs[11, :])
box(ax, "现象",
    "字符类与 spaCy UPOS 在 token-id 之上几乎不再多解释方差。Δ(POS|tok) 中位数只有 0.006–0.013，约为 Δ(prev|tok) 的十分之一；"
    "所有层/ckpt 上 η²(POS) 从未超过 η²(tok)。H(POS|token-id)=0.46 bit / H(POS)=3.8 bit，约 88% 的 POS 信息已被 token-id 决定。",
    NEUT_BG)

ax = fig.add_subplot(gs[12, :2])
style_ax(ax)
ax.plot(x, d_prev, "o-", color=ACCENT, lw=2, ms=5, label="Δ(prev|tok)")
ax.plot(x, d_pos, "o-", color=WARN, lw=2, ms=5, label="Δ(POS|tok)")
ax.plot(x, cc, "o-", color=FG, lw=2, ms=5, label="η²(字符类)")
ax.plot(x, pos, "o--", color=GRAY, lw=2, ms=5, label="η²(POS)")
ax.set_xticks(x); ax.set_xticklabels(layers)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_title("step 10k：POS vs prev 增量", fontsize=11, color=MUTED, loc="left", pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

ax = fig.add_subplot(gs[12, 2:])
ax.axis("off")
ax.text(0, 1.02, "对齐与信息量", fontsize=11, fontweight="650", color=FG, transform=ax.transAxes, va="top")
tbl2 = ax.table(
    cellText=[
        ["one_to_one（词↔token）", "78.0%"],
        ["subword（子词继承）", "10.2%"],
        ["whitespace → SPACE/NEWLINE", "10.6%"],
        ["multi_same / multi_conflict", "0.9% / 0.29%"],
        ["重新分词 id 一致序列", "1924 / 1953"],
    ],
    colLabels=["对齐类别", "token 占比"],
    loc="center", cellLoc="left",
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8.5)
tbl2.scale(1, 1.5)
for (r, c), cell in tbl2.get_celld().items():
    cell.set_edgecolor(STROKE)
    if r == 0:
        cell.set_facecolor("#F0F0ED")
        cell.set_text_props(color=MUTED, fontweight="600")
    else:
        cell.set_facecolor(CARD)

ax = fig.add_subplot(gs[13, :])
box(ax, "解释",
    "字符类是 token-id 的粗化；tok×POS 是 tok 的细分。同一 token 的 POS 变化有限，给定 token 后再看 POS 几乎不增解释力。"
    "形状上 Δ(POS|tok) 与 Δ(prev|tok) 一样在 L4–L8 略高，但幅度小一个数量级。粗粒度（POS/字符类）在 1k 峰值更高，之后被更细信息取代。",
    INFO_BG)

# Todo
ax = fig.add_subplot(gs[14, :])
ax.axis("off")
ax.text(0, 0.5, "Todo", fontsize=14, fontweight="650", color=FG, transform=ax.transAxes, va="center")

ax = fig.add_subplot(gs[15, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.0, 0.02), 0.998, 0.96, boxstyle="round,pad=0.012,rounding_size=0.02",
                            linewidth=1, edgecolor="#FDE68A", facecolor=WARN_BG, transform=ax.transAxes))
ax.text(0.02, 0.90, "下一步可做", transform=ax.transAxes, fontsize=11, fontweight="700", color=FG, va="top")
todos = [
    "1. Δ(POS | bigram)：把 POS 叠在 bigram 上，确认是否带来 n-gram 之外的新信息。",
    "2. 位置桶 / 领域 / 任务因素：research_plan 主序里还没测。",
    "3. 剩余项（L31 ~0.77）：更长程上下文 vs 噪声；可试 induction / 更远 token。",
    "4. 把嵌套链与 POS 对比补进上周层剖面 canvas。",
    "5. 方差加权视角：少数高方差神经元上 POS/字符类占比更高（加权 L0 POS 到 0.46）。",
]
ax.text(0.02, 0.72, "\n".join(todos), transform=ax.transAxes, fontsize=9.5, color=FG, va="top", linespacing=1.55)

ax = fig.add_subplot(gs[16, :])
ax.axis("off")
ax.text(0, 0.6,
        "数据：runs_decomp_trigram/*.npz · runs_decomp_pos/*.npz · frozen_5e6/pos_code.npz · acts/ · T≈2e6 · Nemotron SFT code",
        fontsize=8, color="#8A8A8A", transform=ax.transAxes, va="top")
ax.text(0, 0.1,
        "导出自 Cursor canvas「Olmo3 Factor Decomp Report」（静态等价渲染 · step 10k；非 IDE 内嵌 canvas 像素截图）",
        fontsize=8, color="#8A8A8A", transform=ax.transAxes, va="top")

fig.savefig(out, dpi=160)
print(f"wrote {out}  size={out.stat().st_size/1e6:.2f} MB  cjk_font={cjk!r}")
