"""
Wong 等 (arXiv:2603.13778) 的 CIM 动力学复现 + SK 基准。

CIM 动力学 (论文 Eq. 2):
    dx_i/dt = -x_i^3 + a*x_i + xi * sum_j J_ij x_j + zeta_i
    噪声: <zeta_i(t) zeta_j(t')> = 2 T delta_ij delta(t-t')

读出自旋: s_i = sgn(x_i)
解码能量 (标准约定, 目标 ≈ Parisi 值 -0.763):
    eps(s) = (1/N) * sum_{i<j} J_ij s_i s_j ,  J_ij ~ N(0, 1/N)

两种基准调度 (论文的核心对比):
    增益退火: a 从 a_min 升到 a_max, T 固定(小)
    温度退火: a 固定, T 从 T_high 降到 T_low   (论文主张此路更优)
"""
import numpy as np

# ---------------- 实例与基线 ----------------
def sk_instance(N, seed):
    """SK 模型: 上三角 J ~ N(0, 1/N) 镜像对称, 对角线 0（标准约定，方差严格 1/N）"""
    r = np.random.default_rng(seed)
    J = r.normal(0.0, 1.0 / np.sqrt(N), (N, N))
    J = np.triu(J, 1)
    J = J + J.T
    return J


def energy(J, s):
    """eps(s) = (1/N) sum_{i<j} J_ij s_i s_j ; SK 基态 ≈ -0.763"""
    return (s @ J @ s) / (2.0 * J.shape[0])


def sa_baseline(J, seed, iters=500000, restarts=5):
    """模拟退火基线，拿近似基态能量"""
    r = np.random.default_rng(seed)
    N = J.shape[0]
    best_s, best_e = None, 1e9
    for rs in range(restarts):
        s = r.choice([-1.0, 1.0], N)
        cur = energy(J, s)
        T0 = 0.5
        for k in range(1, iters + 1):
            i = r.integers(N)
            dE = -2.0 * s[i] * (J[i] @ s) / N   # d(eps) 当翻转 i
            if dE <= 0 or r.random() < np.exp(-dE / T0):
                s[i] = -s[i]
                cur += dE
                if cur < best_e:
                    best_e, best_s = cur, s.copy()
            T0 = max(1e-3, 0.5 * (1.0 - k / iters))
    return best_s, best_e


# ---------------- CIM 动力学 ----------------
def cim_run(J, steps, dt=0.02, gain_schedule=None, temp_schedule=None,
            xi=1.0, seed=0, record_every=20):
    """
    跑 CIM。gain_schedule(step)->a, temp_schedule(step)->T。
    欧拉-丸山: x += dt*(-x^3 + a x - xi J x) + sqrt(2 T dt) * N(0,1)
    注意耦合为 -xi*J：最小化 eps = (1/N)Σ J σσ（标准约定，基态≈-0.763）。
    等价于 Wong 论文 Eq.(2) 的 +xi*J 配 H=-(1/2N)ΣJσσ 的约定。
    """
    N = J.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0.0, 0.01, N)
    rec = []
    for t in range(steps):
        a = gain_schedule(t)
        T = temp_schedule(t)
        x = x + dt * (-x ** 3 + a * x - xi * (J @ x))
        x = x + np.sqrt(max(T, 0.0) * 2.0 * dt) * r.standard_normal(N)
        if t % record_every == 0 or t == steps - 1:
            rec.append((t, energy(J, np.sign(x))))
    return np.sign(x), np.array(rec)


def gain_anneal_schedule(a_min, a_max, ramp):
    """a: a_min -> a_max (线性, 到 ramp 步后保持); T 固定由外部给"""
    def sched(t):
        return a_min + (a_max - a_min) * min(1.0, t / ramp)
    return sched


def const_schedule(v):
    return lambda t: v


def temp_anneal_schedule(T_high, T_low, ramp):
    """T: T_high -> T_low (线性)"""
    def sched(t):
        return T_low + (T_high - T_low) * max(0.0, 1.0 - t / ramp)
    return sched
