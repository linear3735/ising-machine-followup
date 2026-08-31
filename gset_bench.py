"""Gset 基准：digCIM 原配方 vs 动量 dSB vs 我们的变体（G1-G3）"""
import numpy as np
import time


def load_gset(path):
    with open(path) as f:
        lines = f.readlines()
    n, m = map(int, lines[0].split())
    A = np.zeros((n, n))
    for ln in lines[1:]:
        parts = ln.split()
        u, v = int(parts[0]) - 1, int(parts[1]) - 1
        w = float(parts[2]) if len(parts) > 2 else 1.0
        A[u, v] += w
        A[v, u] += w
    return A


def cut(A, s):
    m = A.sum() / 2.0
    return m / 2.0 - (s @ A @ s) / 4.0


def digcim_run(A, steps, dt, a, Tinit, seed, record_every=50):
    """论文精确配方: Euler-Maruyama, dx=dt*(a*x - A*sgn(x)) + noise, clip ±1,
    T: Tinit -> 0 线性"""
    N = A.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    best = -1e18
    for t in range(steps):
        T = Tinit * (1.0 - t / max(1, steps - 1))
        x = x + dt * (a * x - (A @ np.sign(x)))
        x = x + np.sqrt(2.0 * max(T, 0) * dt) * r.standard_normal(N)
        x = np.clip(x, -1.0, 1.0)
        if t % record_every == 0 or t == steps - 1:
            c = cut(A, np.sign(x))
            best = max(best, c)
    return np.sign(x), best


def dsb_run(A, steps, dt, a0=1.0, c0=0.01, anneal_frac=0.5, seed=0, record_every=25):
    """dSB: 动量 + 数字化耦合 (J@sgn(x)) + 墙 |x|<=1"""
    N = A.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    y = r.normal(0, 0.01, N)
    T_anneal = max(1, int(steps * anneal_frac))
    best = -1e18
    for t in range(1, steps + 1):
        a = a0 * min(1.0, t / T_anneal)
        y = y + dt * (-(a0 - a) * x - c0 * (A @ np.sign(x)))
        x = x + dt * (a0 * y)
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
        if t % record_every == 0 or t == steps:
            c = cut(A, np.sign(x))
            best = max(best, c)
    return np.sign(x), best


if __name__ == "__main__":
    for g in ("G1", "G2", "G3"):
        A = load_gset(f"{g}.dat")
        m = A.sum() / 2.0
        print(f"=== {g}: N={A.shape[0]}, edges={int(m)} ===")
        known = {"G1": 11624, "G2": 11620, "G3": 11622}[g]
        # digCIM 论文配方
        t0 = time.time()
        c_dig = [digcim_run(A, 5000, 0.03, -10.0, 3.0, 100 + i)[1] for i in range(12)]
        t_dig = time.time() - t0
        print(f"  digCIM 配方(5000步,dt=0.03,T0=3): best={max(c_dig):.0f} "
              f"mean={np.mean(c_dig):.1f} 达标率={(np.array(c_dig) >= known - 1).mean()*100:.0f}% "
              f"[{t_dig:.0f}s]")
        # dSB 动量
        for steps, dt, c0 in ((1000, 0.5, 0.01), (2000, 0.5, 0.01), (1000, 0.5, 0.02)):
            t0 = time.time()
            c_dsb = [dsb_run(A, steps, dt=dt, c0=c0, seed=200 + i)[1] for i in range(12)]
            t_dsb = time.time() - t0
            print(f"  dSB 动量({steps}步,dt={dt},c0={c0}): best={max(c_dsb):.0f} "
                  f"mean={np.mean(c_dsb):.1f} 达标率={(np.array(c_dsb) >= known - 1).mean()*100:.0f}% "
                  f"[{t_dsb:.0f}s]")
        print(f"  已知最优: {known}")
