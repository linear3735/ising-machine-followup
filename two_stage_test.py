"""two_stage_test.py —— 两阶段协议验证 (全退火 -> 低温精修 top-K)

机制问题: 精修 best 的增益来自"温和再热 + 动量越障"(T0=0.05), 还是只是多跑步数?
对照:
  A: 全退火 32x2000 步 (基线, T0=3)
  B1: A + 从 best 终态精修 600 步 T0=0.05 (温和再热)
  B2: A + 从 best 终态继续 600 步 T≈0 (零温延续, 无再热) —— 机制对照
  C: A 但 32x2300 步 (等预算对照: B1 额外 ~1-2% 预算摊给每种子)
判据: best / mean 能量 (SK) 或 cut (Gset), 及 Δbest。
"""
import numpy as np
from cim import sk_instance
from gset_bench import load_gset, cut
from smomentum_fast import smomentum_batch


def eps_sk(J, sgn):
    return (sgn * (sgn @ J)).sum(axis=1) / (2.0 * J.shape[0])


def polish_sk(J, x0s, steps, T0, dt=0.2, gamma=3.0, a=-10.0, seed0=11):
    S = x0s.shape[0]
    N = J.shape[0]
    rngs = [np.random.default_rng(seed0 + s) for s in range(S)]
    x = np.clip(x0s, -1.0, 1.0).copy()
    y = np.zeros_like(x)
    sgn = np.sign(x)
    h = sgn @ J
    best = eps_sk(J, sgn)
    dt_g = 1.0 / (1.0 + dt * gamma)
    for t in range(steps):
        T = max(T0 * (1.0 - t / max(1, steps - 1)), 1e-4)
        ns = np.sqrt(2.0 * T * dt)
        y *= dt_g
        y += (dt * a * x - dt * h) * dt_g
        for s in range(S):
            y[s] += ns * rngs[s].standard_normal(N)
        x += dt * y
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
        sgn = np.sign(x)
        h = sgn @ J
        if t % 100 == 0 or t == steps - 1:
            e = eps_sk(J, sgn)
            best = np.minimum(best, e)
    return eps_sk(J, sgn), best


def run_sk(J, seeds, full_steps=2000, polish_steps=600, T_polish=0.05):
    """返回 (基线best/mean, B1 best/mean, B2 best/mean, C best/mean)"""
    sgn, _ = smomentum_batch(J, full_steps, 0.2, 3.0, -10.0, 3.0, seeds)
    e_full = eps_sk(J, sgn)
    i0 = int(np.argmin(e_full))
    x_best = sgn[i0].astype(np.float64)[None] * 0.999
    # B1: 温和再热精修 (单起点 + 4 噪声种子, 取 best)
    _, b1 = polish_sk(J, np.repeat(x_best, 5, axis=0), polish_steps, T_polish)
    # B2: 零温延续 (T0=1e-4)
    _, b2 = polish_sk(J, np.repeat(x_best, 5, axis=0), polish_steps, 1e-4)
    # C: 等预算: 多跑 ceil(5*600/32)~100 步/种子
    sgnC, _ = smomentum_batch(J, full_steps + 100, 0.2, 3.0, -10.0, 3.0, seeds)
    eC = eps_sk(J, sgnC)
    return (e_full.min(), e_full.mean()), (b1.min(), e_full.mean()), \
           (b2.min(), e_full.mean()), (eC.min(), eC.mean())


def run_gset(A, known, seeds, full_steps=2000, polish_steps=600, T_polish=0.05):
    sgn, _ = smomentum_batch(A, full_steps, 0.2, 3.0, -10.0, 3.0, seeds)
    cuts = np.array([cut(A, s) for s in sgn])
    i0 = int(np.argmax(cuts))
    x_best = sgn[i0].astype(np.float64)[None] * 0.999
    # polish (最大化 cut): 用同一积分器但跟踪 cut
    def pol(x0s, T0):
        S = x0s.shape[0]
        N = A.shape[0]
        rngs = [np.random.default_rng(77 + s) for s in range(S)]
        dt, gamma, a = 0.2, 3.0, -10.0
        dt_g = 1.0 / (1.0 + dt * gamma)
        x = np.clip(x0s, -1, 1).copy()
        y = np.zeros_like(x)
        sgn = np.sign(x)
        h = sgn @ A
        m2 = A.sum() / 4.0
        best = np.array([cut(A, s) for s in sgn])
        for t in range(polish_steps):
            T = max(T0 * (1.0 - t / max(1, polish_steps - 1)), 1e-4)
            ns = np.sqrt(2.0 * T * dt)
            y *= dt_g
            y += (dt * a * x - dt * h) * dt_g
            for s in range(S):
                y[s] += ns * rngs[s].standard_normal(N)
            x += dt * y
            mask = np.abs(x) > 1.0
            x[mask] = np.sign(x[mask])
            y[mask] = 0.0
            sgn = np.sign(x)
            h = sgn @ A
            if t % 100 == 0 or t == polish_steps - 1:
                v = m2 - (sgn * h).sum(axis=1) / 4.0
                best = np.maximum(best, v)
        return best
    b1 = pol(np.repeat(x_best, 5, axis=0), T_polish)
    b2 = pol(np.repeat(x_best, 5, axis=0), 1e-4)
    sgnC, _ = smomentum_batch(A, full_steps + 100, 0.2, 3.0, -10.0, 3.0, seeds)
    eC = np.array([cut(A, s) for s in sgnC])
    return (cuts.max(), cuts.mean()), (b1.max(), cuts.mean()), \
           (b2.max(), cuts.mean()), (eC.max(), eC.mean())


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    SEEDS = list(range(0, 32))
    print("=== SK N=100 (eps, 越小越好; 基线=32x2000全退火) ===")
    print(f"{'实例':>4} {'A best/mean':>16} {'B1再热 best':>12} {'B2零温 best':>12} "
          f"{'C等预算 best/mean':>18}")
    for sid in (101, 102, 200, 201, 202):
        J = sk_instance(100, sid)
        a, b1, b2, c = run_sk(J, SEEDS)
        print(f"{sid:>4} {a[0]:+.4f}/{a[1]:+.4f} {b1[0]:+.4f} {b2[0]:+.4f} "
              f"{c[0]:+.4f}/{c[1]:+.4f}")
    print("\n=== SK N=200 ===")
    for sid in (101, 102):
        J = sk_instance(200, sid)
        a, b1, b2, c = run_sk(J, SEEDS)
        print(f"{sid:>4} {a[0]:+.4f}/{a[1]:+.4f} {b1[0]:+.4f} {b2[0]:+.4f} "
              f"{c[0]:+.4f}/{c[1]:+.4f}")
    print("\n=== Gset (cut, 越大越好) ===")
    print(f"{'实例':>4} {'A best/mean':>16} {'B1再热 best':>12} {'B2零温 best':>12} "
          f"{'C等预算 best/mean':>18}")
    for g in ("G1", "G22"):
        A = load_gset(f"{g}.dat")
        a, b1, b2, c = run_gset(A, None, SEEDS)
        print(f"{g:>4} {a[0]:>6.0f}/{a[1]:>7.1f} {b1[0]:>6.0f} {b2[0]:>6.0f} "
              f"{c[0]:>6.0f}/{c[1]:>7.1f}")
