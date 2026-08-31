"""QPLIB QUBO 攻击：digCIM 配方 vs 贪心热启动，目标 QPLIB_3650 (全局最优 922)"""
import numpy as np
import time


def load_qplib(path):
    lines = open(path).read().splitlines()
    n = int(lines[3].split()[0])
    nquad = int(lines[4].split()[0])
    Q = np.zeros((n, n))
    for ln in lines[5:5 + nquad]:
        parts = ln.split()
        j, i, c = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        Q[i, j] += c
        if i != j:
            Q[j, i] += c
    # 线性项段: 二次块之后
    L = np.zeros(n)
    idx = 5 + nquad + 1                      # 跳过 "default value for linear coefficients"
    nlin = int(lines[idx].split()[0])
    for ln in lines[idx + 1: idx + 1 + nlin]:
        parts = ln.split()
        L[int(parts[0]) - 1] += float(parts[1])
    return Q, L


def qubo_value(Q, L, s):
    """QUBO-QLIB 口径: V(x) = L^T x - (1/2) x^T Q x (官方解 = 920, GT = 922)"""
    x = (s + 1.0) / 2.0
    return float(L @ x - 0.5 * x @ Q @ x)


def greedy_qubo(Q, L, seed):
    r = np.random.default_rng(seed)
    n = Q.shape[0]
    s = np.ones(n)
    for i in r.permutation(n):
        x = (s + 1.0) / 2.0
        delta = -s[i] * (L[i] - (Q @ x)[i])   # 翻转 s_i 对 V 的改变
        if delta > 0:
            s[i] = -s[i]
    return s


def digcim_qubo(Q, L, steps, dt, a, Tinit, seed, warm=False, record_every=100):
    """梯度动力学: F = a*x - (1/4)(Q sgn(x) + Q 1) + (1/2) L"""
    n = Q.shape[0]
    r = np.random.default_rng(seed)
    x = greedy_qubo(Q, L, seed) * 0.5 if warm else r.normal(0, 0.01, n)
    h0 = -0.25 * (Q @ np.ones(n)) + 0.5 * L
    best = -1e18
    for t in range(steps):
        T = Tinit * (1.0 - t / max(1, steps - 1))
        x = x + dt * (a * x - 0.25 * (Q @ np.sign(x)) + h0)
        x = x + np.sqrt(2.0 * T * dt) * r.standard_normal(n)
        x = np.clip(x, -1.0, 1.0)
        if t % record_every == 0 or t == steps - 1:
            best = max(best, qubo_value(Q, L, np.sign(x)))
    return np.sign(x), best


if __name__ == "__main__":
    Q, L = load_qplib("QPLIB_3650.qplib")
    n = Q.shape[0]
    print(f"QPLIB_3650: n={n}, GT = 922, digCIM 论文 = 920 (GT-2)")
    sol = {}
    for ln in open("Q3650.sol").read().splitlines()[1:]:
        p = ln.split()
        if p[0].startswith("b"):
            sol[int(p[0][1:])] = float(p[1])
    xoff = 2 * np.array([sol.get(i + 1, 0.0) for i in range(n)]) - 1
    print(f"口径校验: 官方解 V = {qubo_value(Q, L, xoff):.0f} (应为 920)")
    g_best = max(qubo_value(Q, L, greedy_qubo(Q, L, s)) for s in range(20))
    print(f"纯贪心 20 次 best = {g_best:.0f}")

    for warm in (False, True):
        for steps in (5000, 10000):
            t0 = time.time()
            bests = []
            for sd in range(50):
                _, b = digcim_qubo(Q, L, steps, 0.03, -10.0, 3.0, 800 + sd, warm=warm)
                bests.append(b)
            bests = np.array(bests)
            tag = "digCIM+热启动" if warm else "digCIM"
            print(f"{tag} steps={steps}: "
                  f"best={bests.max():.0f} mean={bests.mean():.1f} "
                  f"达标(>=4280)={(bests>=4280).mean()*100:.0f}% [{time.time()-t0:.0f}s]")
