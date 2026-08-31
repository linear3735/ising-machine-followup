"""最终评估 + 出图：三种调度在 held-out 实例上的对比 + μ 轨迹 + 训练曲线"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cim import sk_instance, energy, sa_baseline, cim_run, \
    gain_anneal_schedule, const_schedule
from closed_loop import rollout_cl, unflat, flat
from learn_schedule import rollout_batch, decode, final_energy

N = 100
SEEDS = 16
STEPS = 2000


def temp_sched(t):
    return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / STEPS)


def run_baselines(Js_val):
    gain_vals, ref_vals, ref_best = [], [], []
    for J in Js_val:
        for sd in range(SEEDS):
            gs = gain_anneal_schedule(0.0, 0.5, STEPS)
            s, _ = cim_run(J, STEPS, gain_schedule=gs,
                           temp_schedule=const_schedule(1e-4), seed=200 + sd)
            gain_vals.append(energy(J, s))
            s, _ = cim_run(J, STEPS, gain_schedule=const_schedule(0.0),
                           temp_schedule=temp_sched, seed=300 + sd)
            ref_vals.append(energy(J, s))
    return np.array(gain_vals), np.array(ref_vals)


def mu_trajectory(J, steps, schedule_mode, seed):
    """计算近零自旋密度 mu(t) 轨迹, 用于对照 Wong 的 effective gap"""
    import cim as C
    r = np.random.default_rng(seed)
    N = J.shape[0]
    x = r.normal(0, 0.01, N)
    traj = []
    for t in range(steps):
        if schedule_mode == "temp":
            T = 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / steps)
            a = 0.0
        elif schedule_mode == "gain":
            T = 1e-4
            a = 0.5 * min(1.0, t / steps)
        x = x + 0.02 * (-x ** 3 + a * x - (J @ x))
        x = x + np.sqrt(2 * T * 0.02) * r.standard_normal(N)
        if t % 25 == 0:
            traj.append((np.abs(x) < 0.01).mean())
    return np.array(traj)


if __name__ == "__main__":
    Js_val = [sk_instance(N, s) for s in range(101, 109)]
    theta_cl = np.load("theta_cl.npy")
    hist_cl = np.load("hist_cl.npy")
    params_cl = unflat(theta_cl)

    print("评估闭环策略 (16 种子)...")
    cl_vals = []
    for sd in range(SEEDS):
        _, final, _ = rollout_cl(Js_val, STEPS, params_cl, seed=777 + sd)
        cl_vals.append(final)
    cl_vals = np.array(cl_vals)

    gain_vals, ref_vals = run_baselines(Js_val)
    sa_ref = np.mean([sa_baseline(J, s)[1] for J in Js_val for s in range(3)])

    print(f"\n=== held-out {len(Js_val)} 实例 x {SEEDS} 种子, N={N} ===")
    for name, v in [("增益退火", gain_vals), ("温度退火(解析)", ref_vals),
                    ("闭环学习(本文)", cl_vals)]:
        print(f"  {name:16s}: mean={v.mean():+.4f} ± {v.std():.4f}, "
              f"best={v.min():+.4f}, 达标率(≤SA+0.02)="
              f"{(v <= sa_ref + 0.02).mean() * 100:.0f}%")
    print(f"  SA 参考: {sa_ref:+.4f}")

    # ---- 图1: 对比条形图 ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["gain annealing", "temperature annealing\n(Wong et al.)",
             "closed-loop policy\n(ours)"]
    vals = [gain_vals, ref_vals, cl_vals]
    ax.bar(names, [v.mean() for v in vals],
           yerr=[v.std() for v in vals], capsize=5,
           color=["#999", "#1f77b4", "#d62728"])
    ax.axhline(sa_ref, color="k", ls="--", lw=1, label=f"SA reference ({sa_ref:.3f})")
    ax.set_ylabel("decoded energy eps (lower is better)")
    ax.set_title(f"SK N={N}, held-out instances x {SEEDS} seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_eval.png", dpi=140)
    plt.close(fig)

    # ---- 图2: 训练曲线 ----
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(hist_cl, lw=1.5, color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train reward (decoded energy)")
    ax.set_title("Closed-loop policy training (ES + group advantage)")
    fig.tight_layout()
    fig.savefig("fig_train.png", dpi=140)
    plt.close(fig)

    # ---- 图3: mu 轨迹 (effective gap 对照) ----
    J0 = Js_val[0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    mu_ref = np.mean([mu_trajectory(J0, STEPS, "temp", s) for s in range(6)], axis=0)
    mu_gain = np.mean([mu_trajectory(J0, STEPS, "gain", s) for s in range(6)], axis=0)
    mu_cl = np.mean([rollout_cl([J0], STEPS, params_cl, seed=900 + s)[2]
                     for s in range(6)], axis=0)
    t = np.arange(len(mu_ref)) * 25
    ax.plot(t, mu_ref, label="temperature annealing (Wong)", color="#1f77b4")
    ax.plot(t, mu_gain, label="gain annealing", color="#999")
    ax.plot(t, mu_cl[:len(mu_ref)], label="closed-loop (ours)", color="#d62728")
    ax.set_xlabel("step")
    ax.set_ylabel("near-zero spin density mu")
    ax.set_title("Effective-gap indicator mu(t): the closed loop keeps spins soft longer")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_mu.png", dpi=140)
    plt.close(fig)
    print("saved: fig_eval.png, fig_train.png, fig_mu.png")
