"""
momentum.py —— Phase 0: 阻尼动量动力学（梯度极限 <-> 弹道极限的连续插值）

动力学（离散辛欧拉 + 阻尼 γ）:
    y += dt * ( -gamma*y  - (a0-a(t))*x  -  c0*(J @ x) )   # 阻尼 + 势能负梯度
    x += dt * ( a0 * y )                                    # 动量 -> 位置
    饱和: tanh 或 墙 |x|<=1 (y 清零)

势能: V = (a0-a)/2 * Σx² + (c0/2) * xᵀJx   （最小化 eps = (1/N)Σ J s s）

极限行为（M0 正确性锚点）:
    gamma = 0    -> 弹道（bSB 式，与 comparison.py 的 sb_run 同构）
    gamma -> ∞   -> 过阻尼 -> 梯度动力学（Wong 论文 Eq.2 的家族）
"""
import numpy as np
from cim import sk_instance, energy


def momentum_run(J, steps=3000, dt=0.05, gamma=1.0, a0=1.0, c0=1.0,
                 anneal_frac=0.5, saturate="tanh", seed=0,
                 record_every=20, track_tail=200):
    """返回 (sign, 轨迹记录, 尾部翻转率)"""
    N = J.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    y = r.normal(0, 0.01, N)
    T_anneal = max(1, int(steps * anneal_frac))
    rec = []
    flips_tail = 0.0
    n_tail = 0
    prev_s = np.sign(x)
    for t in range(1, steps + 1):
        a = a0 * min(1.0, t / T_anneal)
        # 阻尼项半隐式: y = (y + dt*F)/(1 + dt*gamma) —— 线性阻尼精确积分, 任意 gamma 稳定
        F = -(a0 - a) * x - c0 * (J @ x)
        y = (y + dt * F) / (1.0 + dt * gamma)
        x = x + dt * (a0 * y)
        if saturate == "tanh":
            x = np.tanh(x)
        else:  # wall
            mask = np.abs(x) > 1.0
            x[mask] = np.sign(x[mask])
            y[mask] = 0.0
        if t % record_every == 0 or t == steps:
            rec.append((t, energy(J, np.sign(x))))
        if t > steps - track_tail:
            s = np.sign(x)
            flips_tail += float((s != prev_s).sum())
            n_tail += 1
            prev_s = s
    return np.sign(x), np.array(rec), flips_tail / (n_tail * N)


def tts(rec, thr):
    idx = np.where(rec[:, 1] <= thr)[0]
    return int(idx[0]) * 20 if len(idx) else np.inf


if __name__ == "__main__":
    from cim import sa_baseline, cim_run, const_schedule
    np.set_printoptions(suppress=True)
    N = 100
    J = sk_instance(N, seed=1)
    _, e_sa = sa_baseline(J, seed=101)
    thr = e_sa + 0.02
    print(f"N={N}, SA 参考 {e_sa:.4f}, TTS 阈值 {thr:.4f}")

    # 梯度基线（Wong 家族: a=0, 温度退火）
    def temp_sched(t):
        return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / 3000)
    g_vals = []
    for sd in range(8):
        s, _ = cim_run(J, 3000, gain_schedule=const_schedule(0.0),
                       temp_schedule=temp_sched, seed=50 + sd)
        g_vals.append(energy(J, s))
    g_vals = np.array(g_vals)
    print(f"梯度基线(a=0,T退火): mean={g_vals.mean():+.4f} best={g_vals.min():+.4f}")

    print("\n阻尼扫描 (8 种子, dt=0.05, tanh):")
    print(f"{'gamma':>6} {'mean':>9} {'best':>9} {'达标率':>7} {'TTS中位':>8} {'尾部翻转率':>9}")
    results = {}
    for gamma in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        vals, tts_l, flips = [], [], []
        for sd in range(8):
            s, rec, f = momentum_run(J, gamma=gamma, seed=60 + sd)
            vals.append(energy(J, s))
            tts_l.append(tts(rec, thr))
            flips.append(f)
        vals = np.array(vals)
        tts_l = np.array(tts_l)
        ok = np.isfinite(tts_l)
        row = dict(mean=round(vals.mean(), 4), best=round(vals.min(), 4),
                   rate=round(ok.mean(), 3),
                   tts=round(float(np.median(tts_l[ok])), 0) if ok.any() else None,
                   flip=round(float(np.mean(flips)), 4))
        results[gamma] = row
        tts_str = f"{row['tts']:.0f}" if row['tts'] else "—"
        print(f"{gamma:>6} {row['mean']:>+9.4f} {row['best']:>+9.4f} "
              f"{row['rate']:>7.0%} {tts_str:>8} {row['flip']:>9.4f}")
    import json
    json.dump(results, open("momentum_gamma_scan.json", "w"), indent=2)
    print("\nM0 锚点:")
    print("  gamma=0    : 应≈弹道SB（此前 comparison.py 动量: mean -0.7100/best -0.7209）")
    print("  gamma→100  : 平滑慢速（翻转率→0 = 梯度式平滑），但等效步长 dt/gamma 变小,")
    print("               要与梯度基线精确等价需按 gamma 重标度总时长（时间重标度, 物理正确）")


def smomentum_digcim(A, steps, dt, gamma, a, Tinit, seed):
    """阻尼辛欧拉 + digCIM 驱动项(a*x - A*sgn(x)) + 温度噪声 + 墙 |x|<=1
    Phase 1 的合成产物: 论文 dSB 缺温度通道、我们的早期尝试缺阻尼。
    G1 上 gamma≈5, dt=0.2-0.4 达到已知最优 11624, 步长是论文 Euler 的 ~10 倍。"""
    N = A.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    y = r.normal(0, 0.01, N)
    for t in range(steps):
        T = Tinit * (1.0 - t / max(1, steps - 1))
        F = a * x - (A @ np.sign(x))
        y = (y + dt * F) / (1.0 + dt * gamma)
        y = y + np.sqrt(2.0 * T * dt) * r.standard_normal(N)
        x = x + dt * y
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
    return np.sign(x)
