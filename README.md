# Ising Machine Follow-up Research

Independent reproduction and follow-up experiments on the Ising-machine theory of
K. Y. Michael Wong et al.:

- [Phase analysis of Ising machines and their implications on optimization](https://arxiv.org/abs/2507.08533) (digCIM)
- [Optimality and annealing path planning of dynamical analog solvers](https://arxiv.org/abs/2603.13778)

Includes the one-day demo (`demo/`) that started this project, a study on learning
annealing schedules (GRPO-style per-instance group advantages, ES, closed-loop
residual policies driven by the near-zero-spin density — the effective-gap
observable of the paper), a momentum-dynamics study (the paper's explicitly stated
future work), and a Gset / QUBO-QLIB benchmark harness.

## Key results (as of this snapshot)

### 1. SK-model benchmark (held-out instances, 3 instances × 8 seeds)

| N | gain annealing | temperature annealing (Wong et al.) | closed-loop residual (ours) | momentum dynamics (ours) |
|---|---------------|--------------------------------------|-----------------------------|--------------------------|
| 100 | −0.6650 | −0.6893 (TTS 1172) | −0.6899 | **−0.7009 (TTS 56)** |
| 200 | −0.7049 | −0.7271 (TTS 1263) | **−0.7308** | −0.7285 (TTS 47, best −0.7622) |

- Temperature annealing > gain annealing (their central claim) reproduced at both sizes.
- Open-loop schedule learning saturates at the analytic schedule (−0.6999 vs −0.7004) —
  a data-driven endorsement of its optimality within fixed-curve families.
- Closed-loop residual policy (MLP adjusting T multiplicatively around the analytic
  schedule, state = effective-gap density μ, best energy, progress) matches at N=100 and
  gives the best mean with the smallest variance at N=200 (transfer).
- Momentum (Hamiltonian) dynamics: ~20× fewer steps to the 90%-frontier (47–56 vs
  1172–1263), N=200 best −0.7622 ≈ Parisi limit −0.763 — a quantitative instance of the
  "smaller constant factors" conjectured for momentum in arXiv:2603.13778, §VIII.

### 2. Gset MaxCut (digCIM recipe reproduction)

- digCIM's recipe (a=−10, dt=0.03, T: 3→0, clip ±1, 5000 steps) reaches the known
  best on G1 (11624) and G3 (11622). 200-run success probabilities and TTS are
  reported honestly in `report.md`, including a warm-start pilot effect that did not
  survive a larger-sample re-run.

### 3. QUBO-QLIB (QPLIB)

- 23-problem list extracted from the paper (19/23 solved by digCIM; six not at global
  optimum: 3650/3693/3877 at −2, 3832/3838/3850 at −4).
- `qplib_pipeline.py` uses the official `pyqplib` parser (zero convention ambiguity)
  with self-validation against official solution files; multi-core server run planned.

## Layout

- `cim.py` — CIM dynamics + SK instances + SA baseline
- `baseline.py` / `comparison.py` / `tts.py` — gain-vs-temperature reproduction, unified comparison, TTS
- `learn_schedule.py` / `closed_loop.py` — open-loop and closed-loop (residual, μ-driven) schedule learning (ES + GRPO-style group advantages)
- `digcim_closed.py` — closed-loop policy on digCIM dynamics (Gset)
- `gset_bench.py` / `qubo_attack.py` / `qplib_pipeline.py` — Gset and QPLIB harnesses
- `make_report.py` / `gen_pdf.py` — figures + PDF generation
- `report.md` / `onepage.md` / `momentum_plan.md` — reports and the momentum research plan
- `fig_*.png` — main figures
- `demo/` — the one-day Ising-machine demo (simulated bifurcation on MaxCut) + speaker notes

## Reproduce (CPU only, numpy + scipy + pyqplib)

```bash
pip install numpy scipy pyqplib
python baseline.py        # gain vs temperature annealing on SK
python comparison.py      # four-method comparison table
python make_report.py     # aggregated results + figures
python qplib_pipeline.py --problems 3650 --budget 300   # QPLIB single-problem run
```
