"""
一天可完成的 Ising Machine demo：
用「模拟分岔」(Simulated Bifurcation) 解 MaxCut 问题
—— 一个耦合振子动力学系统，退火后自旋从 0 分岔到 ±1。

概念对应：
  连续自旋 x        -> 软自旋（gapless / 无隙相）
  sign(x)           -> 二值自旋（binary / 二值相）
  a(t) 从 0 升到 a0   -> 退火（annealing path）
  x = tanh(x) 饱和   -> 增益饱和 / 数字化（digCIM 的 digitization 精神）
  c0 扫描的 U 型曲线  -> 「调参决定成败」，对应论文里相图共存区的思想

运行：  python demo.py
产出：  fig1_solution.png  fig2_bifurcation.png  fig3_convergence.png  fig4_tuning.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- 问题：随机图上的 MaxCut ----------------
def random_graph(n, p, seed):
    """n 个点、每条边以概率 p 出现的无向图。A[i,j]=1 表示有边。"""
    r = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if r.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A


def sk_instance(n, seed):
    """SK 模型实例：全连接、耦合随机 ±1（自旋玻璃，更难）。"""
    r = np.random.default_rng(seed)
    J = r.choice([-1.0, 1.0], (n, n))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J


def cut(A, s):
    """被切断的边数（或权重和）= (ΣJ - Σ J_ij s_i s_j) / 2，s 取值 ±1"""
    return A.sum() / 4.0 - (s @ A @ s) / 4.0


# ---------------- 基线求解器 ----------------
def greedy(A, seed):
    r = np.random.default_rng(seed)
    n = A.shape[0]
    s = np.ones(n)
    for i in r.permutation(n):
        s[i] = 1.0 if (A[i] @ s) <= 0 else -1.0
    return s


def random_tries(A, seed, tries=1000):
    r = np.random.default_rng(seed)
    n = A.shape[0]
    best_s, best_v = None, -1e9
    for _ in range(tries):
        s = r.choice([-1.0, 1.0], n)
        v = cut(A, s)
        if v > best_v:
            best_v, best_s = v, s
    return best_s


def simulated_annealing(A, seed, iters=100000):
    r = np.random.default_rng(seed)
    n = A.shape[0]
    s = r.choice([-1.0, 1.0], n)
    cur = cut(A, s)
    best_s, best = s.copy(), cur
    T0 = 1.0
    for k in range(1, iters + 1):
        i = r.integers(n)
        dE = -2.0 * s[i] * (A[i] @ s)          # 翻转 i 时能量 E=0.5 s^TAs 的变化
        if dE <= 0 or r.random() < np.exp(-dE / T0):
            s[i] = -s[i]
            cur += -dE / 2.0
            if cur > best:
                best, best_s = cur, s.copy()
        T0 = max(0.01, 1.0 * (1.0 - k / iters))
    return best_s


# ---------------- Ising Machine 本体 ----------------
def bifurcation_coupling(A, frac=0.5, a0=1.0):
    """选耦合强度 c0，让分岔点落在 a = a0*(1-frac)。
    分岔条件：(a0-a) + c0*mu_min = 0  ->  a* = a0 + c0*mu_min。"""
    mu_min = np.linalg.eigvalsh(A).min()
    return frac * a0 / max(1e-9, -mu_min)


def ising_machine(A, seed, steps=4000, dt=0.3, a0=1.0,
                  anneal_frac=0.5, c0=None, saturate=True, record_every=20):
    """
    模拟分岔：连续变量 x（软自旋）+ 动量 y 的耦合振子。
      y += dt * ( -(a0 - a) * x  -  c0 * (A @ x) )   # 势能 V=(a0-a)/2|x|^2+(c0/2)x^TAx 的负梯度
      x += dt * ( a0 * y )                            # 动量 -> 位置
      x  = tanh(x)                                    # 增益饱和（digCIM 式数字化）
    a(t) 在前 anneal_frac 步从 0 线性升到 a0，之后保持。
    """
    n = A.shape[0]
    if c0 is None:
        c0 = 0.5 / max(1.0, A.sum(axis=1).max())
    r = np.random.default_rng(seed)
    x = r.normal(0.0, 0.01, n)
    y = r.normal(0.0, 0.01, n)
    T_anneal = max(1, int(steps * anneal_frac))
    hist = []
    for t in range(1, steps + 1):
        a = a0 * min(1.0, t / T_anneal)
        y = y + dt * (-(a0 - a) * x - c0 * (A @ x))
        x = x + dt * (a0 * y)
        if saturate:
            x = np.tanh(x)
        if t % record_every == 0 or t == steps:
            hist.append((a, cut(A, np.sign(x))))
    return np.sign(x), np.array(hist)


# ---------------- 可视化 ----------------
def plot_solution(A, s, methods, fname):
    n = A.shape[0]
    ang = 2 * np.pi * np.arange(n) / n
    pos = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="0.85", lw=0.6, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], c=np.where(s > 0, "#d62728", "#1f77b4"),
               s=220, zorder=2)
    ax.set_title("MaxCut by the Ising machine\n" + "  |  ".join(methods), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_bifurcation(A, c0, fname, steps=4000, dt=0.3, a0=1.0, T_anneal=2000):
    """看自旋如何从 0 附近分岔出来（软 -> 硬），以及分岔后变成 ±1 两峰。"""
    n = A.shape[0]
    r = np.random.default_rng(7)
    x = r.normal(0.0, 0.01, n)
    y = r.normal(0.0, 0.01, n)
    traj = np.zeros((steps // 10, n))
    snap = {}
    for t in range(1, steps + 1):
        a = a0 * min(1.0, t / T_anneal)
        y = y + dt * (-(a0 - a) * x - c0 * (A @ x))
        x = x + dt * (a0 * y)
        x = np.tanh(x)
        if t % 10 == 0:
            traj[t // 10 - 1] = x
        if t in (200, 1500, 4000):
            snap[t] = (a, x.copy())
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    ax_traj = fig.add_subplot(gs[0, :])
    time_ax = np.arange(steps // 10) * 10
    for i in range(n):
        ax_traj.plot(time_ax, traj[:, i], lw=0.7, alpha=0.7)
    ax_traj.set_xlabel("step")
    ax_traj.set_ylabel("x_i (soft spins)")
    ax_traj.set_title("Bifurcation: spins stay soft (≈0), then harden into ±1 as a ramps up")
    for ax, (t, (a, xv)) in zip([fig.add_subplot(gs[1, k]) for k in range(3)], snap.items()):
        ax.hist(xv, bins=41, range=(-1, 1), color="#1f77b4")
        ax.set_title(f"a = {a:.2f}")
        ax.set_ylim(0, n)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    return fig, snap


def plot_convergence(hist, refs, a_star, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(a_star - 0.05, a_star + 0.05, color="gold", alpha=0.35,
               label="bifurcation")
    ax.plot(hist[:, 0], hist[:, 1], color="#d62728", lw=1.5, label="Ising machine")
    for name, v, c in refs:
        ax.axhline(v, color=c, ls="--", lw=1.2, label=name)
    ax.set_xlabel("annealing parameter a")
    ax.set_ylabel("cut edges")
    ax.set_title("Convergence of the machine vs baselines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_tuning(J, sa_best, fname):
    """耦合强度扫描（SK 实例，短运行）：太弱分岔不出来、太强锁进坏态 -> U 型。"""
    cs = [0.002, 0.004, 0.006, 0.010, 0.020, 0.050, 0.100, 0.200, 0.400]
    vals = []
    for c0 in cs:
        s, _ = ising_machine(J, seed=2, c0=c0, steps=1500, dt=0.2)
        vals.append(cut(J, s))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(cs, vals, "o-", color="#1f77b4")
    ax.axhline(sa_best, color="0.4", ls="--", lw=1.2, label="simulated annealing")
    ax.set_xlabel("coupling strength c0")
    ax.set_ylabel("cut edges")
    ax.set_title("Tuning matters: weak coupling never bifurcates, strong one locks in a bad state")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    # ---- 主演示：稀疏图 ----
    n, p = 40, 0.15
    A = random_graph(n, p, seed=1)
    m = A.sum() / 2.0
    print(f"graph: n={n}, edges={int(m)}")

    c0 = bifurcation_coupling(A, frac=0.5)          # 分岔点放在退火中段
    a_star = 1.0 + c0 * np.linalg.eigvalsh(A).min()  # 理论分岔点
    print(f"coupling c0={c0:.3f} -> bifurcation at a≈{a_star:.2f}")

    s_sb, hist = ising_machine(A, seed=2, c0=c0)
    s_sa = simulated_annealing(A, seed=5)
    s_gr = greedy(A, seed=6)
    s_rd = random_tries(A, seed=8)

    v_sb, v_sa, v_gr, v_rd = (cut(A, s_sb), cut(A, s_sa),
                              cut(A, s_gr), cut(A, s_rd))
    print("cut edges (of", int(m), "):")
    print(f"  Ising machine      : {v_sb:.1f}")
    print(f"  simulated annealing: {v_sa:.1f}")
    print(f"  greedy             : {v_gr:.1f}")
    print(f"  random best/1000   : {v_rd:.1f}")

    methods = [f"Ising machine: {v_sb:.0f}",
               f"SA: {v_sa:.0f}",
               f"greedy: {v_gr:.0f}",
               f"random: {v_rd:.0f}"]
    plot_solution(A, s_sb, methods, "fig1_solution.png")
    fig, snap = plot_bifurcation(A, c0, "fig2_bifurcation.png")
    # 直方图快照：a=0.1 / 0.5 / 1.0 附近
    plot_convergence(hist,
                     [("simulated annealing", v_sa, "0.5"),
                      ("greedy", v_gr, "0.7"),
                      ("random", v_rd, "0.4")],
                     a_star, "fig3_convergence.png")
    plt.close(fig)

    # ---- 调参故事：SK 实例 ----
    J = sk_instance(60, seed=9)
    s_sa_sk = simulated_annealing(J, seed=5, iters=100000)
    v_sa_sk = cut(J, s_sa_sk)
    print(f"SK instance n=60: SA best = {v_sa_sk:.1f}")
    plot_tuning(J, v_sa_sk, "fig4_tuning.png")

    print("saved: fig1_solution.png, fig2_bifurcation.png, "
          "fig3_convergence.png, fig4_tuning.png")
