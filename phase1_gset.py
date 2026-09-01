"""Phase 1: G1 诊断 —— 动量在哪个阶段输给梯度 digCIM"""
import numpy as np
import time
from gset_bench import load_gset, cut
from momentum import momentum_run
from cim import energy as sk_energy


def momentum_gset(A, steps, dt, gamma, a0, c0, seed, record_every=100):
    """Gset 版动量: 最小化 s^T A s (MaxCut)。返回 (最终cut, mu轨迹, 动能轨迹)"""
    N = A.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    y = r.normal(0, 0.01, N)
    T_anneal = max(1, int(steps * 0.5))
    mu_traj, kin_traj, cut_traj = [], [], []
    for t in range(1, steps + 1):
        a = a0 * min(1.0, t / T_anneal)
        F = -(a0 - a) * x - c0 * (A @ x)
        y = (y + dt * F) / (1.0 + dt * gamma)
        x = x + dt * (a0 * y)
        x = np.tanh(x)
        if t % record_every == 0 or t == steps:
            mu_traj.append(float((np.abs(x) < 0.01).mean()))
            kin_traj.append(float((y ** 2).mean()))
            cut_traj.append(cut(A, np.sign(x)))
    return cut(A, np.sign(x)), np.array(mu_traj), np.array(kin_traj), np.array(cut_traj)


def digcim_gset(A, steps, dt, a, Tinit, seed, record_every=100):
    """论文配方梯度 digCIM"""
    N = A.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    mu_traj, cut_traj = [], []
    for t in range(steps):
        T = Tinit * (1.0 - t / max(1, steps - 1))
        x = x + dt * (a * x - (A @ np.sign(x)))
        x = x + np.sqrt(2.0 * T * dt) * r.standard_normal(N)
        x = np.clip(x, -1.0, 1.0)
        if t % record_every == 0 or t == steps - 1:
            mu_traj.append(float((np.abs(x) < 0.01).mean()))
            cut_traj.append(cut(A, np.sign(x)))
    return cut(A, np.sign(x)), np.array(mu_traj), np.array(cut_traj)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    A = load_gset("G1.dat")
    lam_max = np.linalg.eigvalsh(A).max()
    c0 = 2.0 / lam_max          # 归一化耦合, 与 SK 的 c0=1 同尺度
    print(f"G1: N={A.shape[0]}, λ_max={lam_max:.1f}, 归一化 c0={c0:.4f}, 已知最优 11624")

    t0 = time.time()
    cb, _, _ = digcim_gset(A, 5000, 0.03, -10.0, 3.0, 100)
    print(f"梯度 digCIM 基线: best={cb:.0f} [{time.time()-t0:.0f}s]")

    print(f"{'gamma':>6} {'mean':>7} {'best':>7} {'末段mu':>8} {'末段动能':>9} {'翻转率':>7}")
    for gamma in (0.0, 0.3, 1.0, 3.0, 10.0):
        vals, mu_end, kin_end = [], [], []
        for sd in range(8):
            c, mu, kin, _ = momentum_gset(A, 5000, 0.05, gamma, 1.0, c0, 200 + sd)
            vals.append(c)
            mu_end.append(mu[-10:].mean())
            kin_end.append(kin[-10:].mean())
        vals = np.array(vals)
        print(f"{gamma:>6} {vals.mean():>7.1f} {vals.max():>7.0f} "
              f"{np.mean(mu_end):>8.4f} {np.mean(kin_end):>9.3f}")
    print(f"已知最优 11624, 梯度基线 {cb}")
