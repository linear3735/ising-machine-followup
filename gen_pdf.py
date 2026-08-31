"""生成给王教授的 Gset/SK 复现与跟进实验 PDF"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image)

W, H = A4
styles = getSampleStyleSheet()
title = ParagraphStyle("title", parent=styles["Title"], fontSize=17,
                       spaceAfter=4)
sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10.5,
                     textColor=colors.grey, alignment=1, spaceAfter=14)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13,
                    spaceBefore=12, spaceAfter=6)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5,
                      leading=15, spaceAfter=6)
cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=9.5, leading=12)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")


def T(data, widths, header=None):
    rows = [[Paragraph(c, cellb if r == 0 else cell) for c in row]
            for r, row in enumerate(([header] if header else []) + data)]
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.94, 0.97)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


doc = SimpleDocTemplate("wong_followup.pdf", pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Follow-up experiments on Ising-machine benchmarks")
S = []
S.append(Paragraph("Follow-up experiments on Ising-machine optimization benchmarks", title))
S.append(Paragraph("Notes prepared for Prof. K. Y. Michael Wong &nbsp;·&nbsp; "
                   "Heming Shi (MSc Physics, Scientific Computing, HKUST) &nbsp;·&nbsp; "
                   "based on arXiv:2603.13778 and arXiv:2507.08533", sub))

S.append(Paragraph("1. Purpose", h1))
S.append(Paragraph(
    "I independently reproduced the digCIM experiments reported in your papers and ran a set of "
    "follow-ups. This note reports, transparently, which results held up (the reproductions and the "
    "momentum-dynamics findings) and which did not (a warm-start pilot that did not survive a "
    "larger-sample re-run). All experiments use numpy on a laptop CPU; scripts are available on request.", body))

S.append(Paragraph("2. Gset MaxCut: reproduction of digCIM", h1))
S.append(Paragraph(
    "Dynamics follow the digCIM recipe (arXiv:2507.08533, SM Sec. 15): "
    "<i>dx = dt·(a·x − A·sgn(x)) + noise</i>, Euler–Maruyama, a = −10, dt = 0.03, "
    "T annealed linearly 3 → 0, clip x to ±1, 5000 steps. Success is defined as the final state "
    "reaching the published best-known cut; TTS = steps·log(0.01)/log(1−Ps).", body))
S.append(T([["Gset", "best-known", "init", "best found", "Ps (200 runs)", "TTS (steps)"],
            ["G1", "11624", "noise", "11624 ✓", "26.0%", "76 471"],
            ["G1", "11624", "greedy warm", "11624 ✓", "27.0%", "73 165"],
            ["G3", "11622", "noise", "11622 ✓", "11.5%", "188 478"],
            ["G3", "11622", "greedy warm", "11622 ✓", "9.0%", "244 149"]],
           [4.2 * cm, 2.6 * cm, 3.4 * cm, 2.6 * cm, 3.0 * cm, 2.6 * cm]))
S.append(Spacer(1, 4))
S.append(Paragraph(
    "<b>Honest note on the warm-start experiment.</b> A first pilot (100 runs) suggested that a greedy "
    "warm start reduced TTS on G1 by ~26%. A 200-run confirmation did not reproduce this: −4% on G1 "
    "and a slight loss on G3 — within binomial sampling noise. I therefore do not claim a warm-start "
    "improvement; I report both runs above so the fluctuation is visible. (A lesson I took from it: "
    "Ps-based TTS estimates need ≥ several hundred runs before being quoted.)", body))
S.append(Image("fig_gset.png", width=16 * cm, height=6.1 * cm))
S.append(Paragraph(
    "Two further negatives, both consistent with your papers: (i) a grid sweep over (T₀, cooling "
    "exponent, steps) found the linear schedule near-optimal — independently confirming your "
    "parameter search; (ii) a momentum (dSB-style) variant was worse than digCIM on G1–G3, matching "
    "your Table 9 (dSB only wins on larger instances).", body))

S.append(Paragraph("3. SK model: the four-method comparison (held-out instances)", h1))
S.append(Paragraph(
    "On the SK model (J ~ N(0, 1/N), 3 held-out instances × 8 seeds) I compared gain annealing, "
    "your temperature-annealing schedule, a closed-loop residual policy (a small MLP that adjusts T "
    "multiplicatively around your analytic schedule, with the near-zero spin density μ — your "
    "effective-gap observable — as its state input; trained at N=100 via ES with per-instance group "
    "advantages), and a momentum (Hamiltonian) dynamics — the regime your arXiv:2603.13778 flags as "
    "future work.", body))
S.append(T([["N", "gain ann.", "temp ann. (yours)", "closed-loop (ours)", "momentum (ours)"],
            ["100", "−0.6650", "−0.6893 (TTS 1172)", "−0.6899", "−0.7009 (TTS 56)"],
            ["200", "−0.7049", "−0.7271 (TTS 1263)", "−0.7308", "−0.7285 (TTS 47)"]],
           [1.5 * cm, 2.6 * cm, 4.2 * cm, 3.8 * cm, 4.0 * cm]))
S.append(Spacer(1, 4))
S.append(Paragraph(
    "<b>Findings.</b> (i) Temperature annealing > gain annealing at both sizes — your central claim "
    "reproduced. (ii) Open-loop schedule learning saturated at your analytic schedule (−0.6999 vs "
    "−0.7004) — a data-driven endorsement of its optimality within fixed-curve families. (iii) The "
    "closed-loop residual policy matches at N=100 and, transferred to N=200, gives the best mean "
    "(−0.7308) with markedly smaller variance. (iv) Momentum dynamics converge ~20× faster "
    "(47–56 vs 1172–1263 steps to the 90%-frontier level) and reach −0.7622 at N=200, close to the "
    "Parisi limit −0.763 — a quantitative instance of the smaller-constant-factor behavior your "
    "paper conjectured for Hamiltonian dynamics.", body))
S.append(Image("fig_compare.png", width=15.6 * cm, height=6.4 * cm))
S.append(Image("fig_tts.png", width=15.6 * cm, height=5.7 * cm))

S.append(Paragraph("4. Next step: the QUBO-QLIB suite", h1))
S.append(Paragraph(
    "I extracted the full 23-problem list and digCIM's per-problem results from your SM Table 8 "
    "(19/23 solved; the six not at global optimum: 3650/3693/3877 at −2, and 3832/3838/3850 at −4). "
    "A pipeline (download → parse → self-validate against the official solution files → greedy + "
    "digCIM multi-start under a 1-hour budget) is ready at the code level; running the full suite "
    "needs a multi-core server, which I plan to rent. If all six are closed, that would be 23/23 "
    "under your protocol — I will report whatever the runs actually give.", body))

S.append(Paragraph("5. What I would like to discuss", h1))
S.append(Paragraph(
    "① Whether the momentum advantage extends to the effective-gap picture (μ as a schedule driver); "
    "② whether an RL-trained schedule (GRPO-style per-instance advantages) is a sensible direction "
    "for the QUBO-QLIB re-run; ③ how the project should balance the analytic (your framework) and "
    "data-driven (my background) sides over the semester.", body))

doc.build(S)
print("PDF saved: wong_followup.pdf")
