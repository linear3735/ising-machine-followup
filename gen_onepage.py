"""一页卡: onepage.pdf —— 16:00 会议用单页结果卡 (A4)"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
CJK = "STSong-Light"

W, H = A4
styles = getSampleStyleSheet()
title = ParagraphStyle("title", parent=styles["Title"], fontSize=15,
                       spaceAfter=3, fontName=CJK)
sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9,
                     textColor=colors.grey, alignment=1, spaceAfter=8,
                     fontName=CJK)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=10.5,
                    spaceBefore=7, spaceAfter=3,
                    textColor=colors.HexColor("#1a3a6b"), fontName=CJK)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.6, leading=11.5,
                      spaceAfter=4, fontName=CJK)
cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=10)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")


def T(data, widths):
    rows = [[Paragraph(c, cellb if r == 0 else cell) for c in row]
            for r, row in enumerate(data)]
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.94, 0.97)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


doc = SimpleDocTemplate("onepage.pdf", pagesize=A4,
                        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                        title="One-page result card")
S = []
S.append(Paragraph("复现 + 跟进 Wong 等 (arXiv:2603.13778) —— 一页结果卡", title))
S.append(Paragraph("时和鸣（HKUST MSc Physics, Scientific Computing）· 基于 arXiv:2603.13778 与 arXiv:2507.08533 · 2026-09-01", sub))

S.append(Paragraph("一句话：复现了论文核心结论（温度退火 &gt; 增益退火），并量化两个增量——① 闭环残差调度（μ 驱动的 ±30% 修正，N=200 迁移后均值最优、方差最小）；② 动量动力学（论文 future work）：TTS 快 20-25×，N=200 最优 −0.7622 ≈ Parisi −0.763。", body))

S.append(Paragraph("主表（held-out 3 实例 × 8 种子）", h1))
S.append(T([["N", "增益退火", "温度退火(论文)", "闭环残差(我)", "动量(我)"],
            ["100", "−0.6650", "−0.6893 (TTS 1172)", "−0.6899", "−0.7009 (TTS 56)"],
            ["200", "−0.7049", "−0.7271 (TTS 1263)", "−0.7308", "−0.7285 (TTS 47, best −0.7622)"]],
           [1.2 * cm, 2.2 * cm, 4.4 * cm, 3.6 * cm, 5.2 * cm]))
S.append(Spacer(1, 2))

S.append(Paragraph("Gset G1：阻尼动量 × digCIM（诚实口径，64 种子）", h1))
S.append(T([["配方", "步数", "dt", "best", "Ps", "TTS(步)"],
            ["digCIM（论文配方）", "5000", "0.03", "11624 ✓", "26.6%", "~74 500"],
            ["digCIM（同预算）", "2000", "0.03", "11624 ✓", "6.2%", "~133 000"],
            ["smomentum (γ=3)", "2000", "0.20", "11624 ✓", "23.4%", "~34 500"]],
           [4.0 * cm, 1.6 * cm, 1.4 * cm, 2.0 * cm, 2.2 * cm, 2.4 * cm]))
S.append(Spacer(1, 2))
S.append(Paragraph("smomentum（阻尼辛欧拉 × digCIM 驱动 × 温度 × 墙）在 2000 步、dt=0.2（≈7× 论文 Euler 0.03）可达已知最优 11624；同预算成功率是 digCIM 的 3.8×，与论文 5000 步配方相当且步数少 60%。稳定窗窄（γ∈[3,5], dt≤0.25），与论文“dSB 仅大实例胜出”一致 → 下一步首选 G11-G21。", body))

S.append(Paragraph("方法要点（30 秒）", h1))
S.append(Paragraph("① CIM 梯度动力学（Eq.2，欧拉-丸山 dt=0.02，eps=(1/N)ΣJσσ，J~N(0,1/N)）；② 闭环：T(t)=T_analytic(t)·exp(0.3·tanh(MLP(μ,eps,f,offset)))，μ=近零自旋密度=论文 effective-gap 可观测量；③ 训练：ES + GRPO 式实例内分组优势 + 对偶采样，N=100 训练/N=200 迁移；④ 动量：SB 式 y+=dt(−(a0−a)x−c0Jx), x+=dt·a0y, tanh/墙饱和。", body))

S.append(Paragraph("三点发现", h1))
S.append(Paragraph("1. 复现：温度退火 &gt; 增益退火（N=100/200 均成立）。2. 背书：开环学习调度收敛到解析解附近（−0.6999 vs −0.7004）→ 解析调度在固定曲线家族内近最优。3. 补空白：动量动力学 TTS 约 20-25× 加速、N=200 最优逼近 Parisi 极限。", body))

S.append(Paragraph("与 IEEE RL-assisted Annealing (Yasudo 等) 的关系", h1))
S.append(Paragraph("他们侧重多 GPU 工程化 RL 辅助模拟退火解 QUBO；本文侧重调度方法论 + 论文 future work 的动量动力学验证，两条线互补。", body))

doc.build(S)
print("saved: onepage.pdf")
