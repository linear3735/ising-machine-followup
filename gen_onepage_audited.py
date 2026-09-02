"""修正版一页卡 PDF (审计后内容)"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
CJK = "STSong-Light"
styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=14, spaceAfter=3, fontName=CJK)
sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey,
                     alignment=1, spaceAfter=8, fontName=CJK)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=10.5, spaceBefore=6,
                    spaceAfter=3, textColor=colors.HexColor("#1a3a6b"), fontName=CJK)
body = ParagraphStyle("b", parent=styles["Normal"], fontSize=8.4, leading=11, spaceAfter=3,
                      fontName=CJK)
cell = ParagraphStyle("c", parent=styles["Normal"], fontSize=7.8, leading=9.5, fontName=CJK)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")

def T(data, widths):
    rows = [[Paragraph(c, cellb if r == 0 else cell) for c in row] for r, row in enumerate(data)]
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                           ("BACKGROUND", (0,0), (-1,0), colors.Color(0.92,0.94,0.97)),
                           ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                           ("TOPPADDING", (0,0), (-1,-1), 2),
                           ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    return t

doc = SimpleDocTemplate("onepage_audited.pdf", pagesize=A4,
                        leftMargin=1.6*cm, rightMargin=1.6*cm,
                        topMargin=1.3*cm, bottomMargin=1.3*cm)
S = []
S.append(Paragraph("复现 + 压力测试：Wong 等 (arXiv:2603.13778 / 2507.08533) —— 修正版一页卡", title))
S.append(Paragraph("时和鸣 · 2026-09-02 · 全部数字经审计（等物理时间 / held-out / 超参先定），可答追问", sub))

S.append(Paragraph("一句话", h1))
S.append(Paragraph("独立复现：温度退火 &gt; 增益退火 ✓、digCIM G1=11624 ✓；7 种独立方法在同等资源协议下"
                   "均未能击败解析温度调度；对论文 future work“动量更小常数因子”猜想给出第一次数值检验："
                   "<b>速度-可靠性权衡，非支配</b>。", body))

S.append(Paragraph("1. 复现（held-out, 同协议）", h1))
S.append(T([["N", "增益退火", "温度退火(论文)"],
            ["100", "−0.6650", "−0.6893"],
            ["200", "−0.7049", "−0.7271"]], [2.5*cm, 4.5*cm, 4.5*cm]))
S.append(Paragraph("digCIM 配方 G1: best 11624 = 已知最优（64 种子 Ps≈27%）。", body))

S.append(Paragraph("2. 学习背书（数据式确认）", h1))
S.append(Paragraph("开环学习调度收敛到解析调度附近（−0.6999 vs −0.7004）；闭环 μ-策略（用论文 "
                   "effective-gap 可观测量做状态）与温度退火无显著差异（192 配对 p=0.94）"
                   "→ 解析调度在可学习家族内近最优。", body))

S.append(Paragraph("3. 动量猜想检验（新内容：等物理时间、held-out、γ 在开发实例先定）", h1))
S.append(T([["口径", "结果"],
            ["成功跑的物理时间", "快 1.3-2.9× (γ=0.1-0.3)"],
            ["达标率 (60tu 内)", "系统性低 20-40pp"],
            ["期望 TTS", "约半数实例被低达标率抹平"],
            ["γ 敏感性", "0.1-1.0 选参在噪声内；权衡随 γ 大改"]], [5.0*cm, 8.5*cm]))
S.append(Paragraph("→ “更小常数因子”成立但限定：速度-可靠性权衡；解释 dSB 为何只在特定规模/参数胜出。", body))

S.append(Paragraph("4. 系统性负结果（6-7 个独立实验，全部文档化）", h1))
S.append(Paragraph("动量变体 / 闭环学习 / 重启策略 / 草案初始 / flow 潜空间草案 → 同等预算下全部无杠杆。"
                   "结论：充分退火的动力学自足；预算分配 &gt; 初始质量。", body))

S.append(Paragraph("5. 基础设施（可复用）", h1))
S.append(Paragraph("QUBO-QLIB 口径解决（½×二次项 + .sol 命名 b_k↔k−1），6/6 官方解精确复算 objvar"
                   "（3650=922, 3693=1154, …），旧“4334”口径废弃；批量化+稀疏求解器同结果 2-13× 墙钟；"
                   "修正公开数据 G16=3052/G17=3047；G11-G21 为带符号权重 ±1。", body))

S.append(Paragraph("6. 下一步（征求方向）", h1))
S.append(Paragraph("① QUBO-QLIB 服务器跑分：digCIM 自败 6 题（差 2-4 个单位）——唯一合法超车战场；"
                   "② 动量权衡推到 N=10⁴（论文规模）验证随 N 的变化。", body))
doc.build(S)
print("saved onepage_audited.pdf")
