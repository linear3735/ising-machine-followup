"""smomentum_restart.py —— 草案头移植到 smomentum×Gset 的规则版击杀测试

预算固定 (2000 步/种子, 重启不增加步数), 重启动作分 elitist/diverse 两型:
  elitist : x <- 0.9*sign(best_x) + 0.1*N(0,1)   (从 incumbent 再热)
  diverse : x <- N(0, 0.01)                      (全随机重熔, 与 flow 草案器同语义)
  reheat  : T = Tinit 持续 300 步 (再热脉冲), 然后回到全局线性调度
触发:
  fixed : t in {700, 1400}
  stall : 300 步无改进 且 mu<0.02 且 t<1700 且 重启数<3
判据: 全程 running best (公平口径) + 终态; Ps(>=known)。
"""
import numpy as np
import scipy.sparse as sp
import time

from gset_bench import load_gset, cut


def smomentum_restart(A, steps, dt, gamma, a, Tinit, seeds, mode="base",
                      reheat_len=300, stall_windows=3, max_restart=3):
    """mode: base / fixed_elite / fixed_diverse / stall_elite / stall_diverse"""
    S = len(seeds)
    N = A.shape[0]
    csr = sp.csr_matrix(np.asarray(A))
    rngs = [np.random.default_rng(sd) for sd in seeds]
    x = np.array([r.normal(0, 0.01, N) for r in rngs])
    y = np.array([r.normal(0, 0.01, N) for r in rngs])
    sgn = np.sign(x)
    h = np.asarray(csr @ sgn.T)
    m2 = csr.sum() / 4.0
    best_cut = np.array([cut(A, s) for s in sgn])
    best_x = x.copy()
    stall_cnt = np.zeros(S, int)
    n_rest = np.zeros(S, int)
    reheat = np.zeros(S, int)
    dt_g = 1.0 / (1.0 + dt * gamma)
    a_dt = dt * a
    fix_times = (700, 1400)
    for t in range(steps):
        T_glob = Tinit * (1.0 - t / max(1, steps - 1))
        T = np.where(reheat > 0, Tinit, T_glob)
        ns = np.sqrt(2.0 * T * dt)
        y *= dt_g
        y += (a_dt * x - dt * h.T) * dt_g
        for s in range(S):
            y[s] += ns[s] * rngs[s].standard_normal(N)
        x += dt * y
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
        new_sgn = np.sign(x)
        h = np.asarray(csr @ new_sgn.T)
        sgn = new_sgn
        reheat = np.maximum(reheat - 1, 0)
        if t % 100 == 0 or t == steps - 1:
            cuts = m2 - (sgn * h.T).sum(axis=1) / 4.0
            imp = cuts > best_cut
            best_cut = np.maximum(best_cut, cuts)
            best_x[imp] = x[imp]
            stall_cnt = np.where(imp, 0, stall_cnt + 1)
            if mode.startswith("fixed") and t in fix_times:
                trig = np.ones(S, bool)
            elif mode.startswith("stall"):
                mu = (np.abs(x) < 0.01).mean(axis=1)
                trig = (stall_cnt >= stall_windows) & (mu < 0.02) \
                       & (t < 1700) & (n_rest < max_restart)
            else:
                trig = np.zeros(S, bool)
            if trig.any():
                idx = np.where(trig)[0]
                if mode.endswith("elite"):
                    x[idx] = 0.9 * np.sign(best_x[idx]) \
                             + 0.1 * np.array([rngs[s].normal(0, 1, N) for s in idx])
                else:
                    x[idx] = np.array([rngs[s].normal(0, 0.01, N) for s in idx])
                y[idx] = np.array([rngs[s].normal(0, 0.01, N) for s in idx])
                y[idx] = 0.0
                new_sgn = np.sign(x)
                h = np.asarray(csr @ new_sgn.T)
                sgn = new_sgn
                reheat[idx] = reheat_len
                n_rest[idx] += 1
                stall_cnt[idx] = 0
    final = m2 - (sgn * h.T).sum(axis=1) / 4.0
    return best_cut, final, n_rest


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    SEEDS = list(range(2000, 2032))  # 32 种子
    STEPS, DT, GAMMA, A_P, T0 = 2000, 0.2, 3.0, -10.0, 3.0
    insts = ["G1", "G11", "G22"]
    known = {"G1": 11624, "G11": 564, "G22": 13359}
    modes = ["base", "fixed_elite", "fixed_diverse", "stall_elite", "stall_diverse"]
    print(f"{'实例':>4} {'模式':>14} {'best(全程)':>10} {'mean(全程)':>11} "
          f"{'Ps':>6} {'final_mean':>11} {'重启/种子':>7}")
    print("-" * 80)
    for g in insts:
        A = load_gset(f"{g}.dat")
        k = known[g]
        for mode in modes:
            t0 = time.time()
            best, final, nr = smomentum_restart(A, STEPS, DT, GAMMA, A_P, T0,
                                                SEEDS, mode=mode)
            ps = (best >= k).mean()
            print(f"{g:>4} {mode:>14} {best.max():>10.0f} {best.mean():>11.1f} "
                  f"{ps:>6.0%} {final.mean():>11.1f} {nr.mean():>6.1f} "
                  f"[{time.time()-t0:.1f}s]")
        print()
