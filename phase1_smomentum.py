"""Phase 1 复验: smomentum_digcim 在 G1 上是否稳定打到已知最优 11624。

复验 momentum_plan.md 的合成突破主张:
  阻尼辛欧拉 × digCIM 驱动项(a*x - A*sgn(x)) × 温度退火 × 墙 |x|<=1
  → G1 best = 11624 = 已知最优, 16 种子 0% 失败率, γ∈[3,8]×dt∈[0.2,0.4]

口径: 终态 cut + 全程 best cut 双口径; 步数 2000 (论文 5000 的 40%),
步长 dt=0.2-0.4 (论文 Euler 0.03 的 7-13 倍), a=-10, Tinit=3 (论文配方)。
"""
import numpy as np
import time
from gset_bench import load_gset, cut
from momentum import smomentum_digcim


def run_smomentum(A, steps, dt, gamma, a, Tinit, seed):
    """在 smomentum_digcim 基础上额外跟踪全程 best cut。"""
    N = A.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    y = r.normal(0, 0.01, N)
    best = -1e18
    for t in range(steps):
        T = Tinit * (1.0 - t / max(1, steps - 1))
        F = a * x - (A @ np.sign(x))
        y = (y + dt * F) / (1.0 + dt * gamma)
        y = y + np.sqrt(2.0 * T * dt) * r.standard_normal(N)
        x = x + dt * y
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
        if t % 50 == 0 or t == steps - 1:
            best = max(best, cut(A, np.sign(x)))
    return cut(A, np.sign(x)), best


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    A = load_gset("G1.dat")
    known = 11624
    print(f"G1: N={A.shape[0]}, 已知最优 {known}")

    gammas = [3.0, 5.0, 8.0]
    dts = [0.2, 0.3, 0.4]
    seeds = list(range(16))
    a, Tinit, steps = -10.0, 3.0, 2000

    t0 = time.time()
    print(f"\n{'gamma':>5} {'dt':>4} {'终态best':>9} {'终态mean':>9} "
          f"{'终态达标率':>9} {'全程best':>9} {'全程达标率':>9}")
    rows = []
    for gamma in gammas:
        for dt in dts:
            fin, run_best = [], []
            for sd in seeds:
                f, rb = run_smomentum(A, steps, dt, gamma, a, Tinit, 1000 + sd)
                fin.append(f)
                run_best.append(rb)
            fin = np.array(fin)
            rb = np.array(run_best)
            fin_rate = (fin >= known).mean()
            rb_rate = (rb >= known).mean()
            rows.append(dict(gamma=gamma, dt=dt,
                             fin_best=int(fin.max()), fin_mean=round(float(fin.mean()), 1),
                             fin_rate=round(float(fin_rate), 3),
                             rb_best=int(rb.max()), rb_rate=round(float(rb_rate), 3)))
            print(f"{gamma:>5.1f} {dt:>4.1f} {int(fin.max()):>9} {float(fin.mean()):>9.1f} "
                  f"{fin_rate:>9.0%} {int(rb.max()):>9} {rb_rate:>9.0%}")
    print(f"\n总耗时 {time.time()-t0:.1f}s")
    import json
    json.dump(rows, open("smomentum_g1_grid.json", "w"), indent=2)
    print("已写入 smomentum_g1_grid.json")
