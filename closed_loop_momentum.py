"""closed_loop_momentum.py —— 在动量动力学上做闭环调度训练（v2 的 20x 版）

动机: 动量动力学 SK TTS 47-56 步 vs 梯度 1172-1263 步 -> 训练 rollout
同样应该用动量动力学, 300 步代替 1500 步, 训练墙钟 ~5x, 且学到的调度
直接部署在动量系统上（我们的主线）。结构同 closed_loop_v2:
残差控制 + MLP + ES/GRPO 组内优势 + 对偶采样 + Adam。
"""
import numpy as np
import time
from closed_loop import init_mlp, warm_start, flat, unflat, mlp_apply, T_MIN
from closed_loop_v2 import rollout_cl_batch as _unused  # noqa


def rollout_mom(Js, steps, thetas, seed, dt=0.05, gamma=0.3, a0=1.0, c0=1.0,
                control_every=5):
    """动量动力学闭环 rollout: y+=dt(−γy−(a0−a)x−c0Jx); x+=dt·a0·y; tanh;
    T(t)=base_T(t)·exp(0.3·tanh(MLP(s))), 噪声加在 y 上 (√(2T·dt)ξ)。"""
    K = len(thetas)
    B = len(Js)
    N = Js[0].shape[0]
    Jbig = np.repeat(np.stack(Js), K, axis=0)
    r = np.random.default_rng(seed)
    x = r.normal(0.0, 0.01, (K * B, N))
    y = r.normal(0.0, 0.01, (K * B, N))
    offset = np.zeros((K * B, 1))
    best_eps = np.zeros((K * B, 1))
    T_anneal = max(1, int(steps * 0.5))
    dt_g = 1.0 / (1.0 + dt * gamma)
    for t in range(1, steps + 1):
        frac = (t - 1) / max(1, steps - 1)
        a = a0 * min(1.0, t / T_anneal)
        base_T = 0.005 * (1.0 - frac) + T_MIN            # 参考: 线性温度
        T = base_T * np.exp(offset)
        Jx = np.matmul(Jbig, x[:, :, None])[:, :, 0]
        F = -(a0 - a) * x - c0 * Jx
        y = (y + dt * F) * dt_g
        y = y + np.sqrt(2.0 * T * dt) * r.standard_normal((K * B, N))
        x = x + dt * (a0 * y)
        x = np.tanh(x)
        if t % control_every == 0:
            sgn = np.sign(x)
            eps = np.matmul(sgn[:, None, :], np.matmul(Jbig, sgn[:, :, None]))[:, 0, 0] / (2.0 * N)
            best_eps = np.minimum(best_eps, eps[:, None])
            mu = (np.abs(x) < 0.01).mean(axis=1)[:, None]
            for k in range(K):
                p = unflat(thetas[k])
                st = np.concatenate([mu[k * B:(k + 1) * B], best_eps[k * B:(k + 1) * B],
                                     np.full((B, 1), frac), offset[k * B:(k + 1) * B]], axis=1)
                offset[k * B:(k + 1) * B] = 0.3 * np.tanh(mlp_apply(p, st))
    sgn = np.sign(x)
    final = np.matmul(sgn[:, None, :], np.matmul(Jbig, sgn[:, :, None]))[:, 0, 0] / (2.0 * N)
    return final


def train_mom(Js_train, steps=300, epochs=150, K=12, sigma=0.1, lr=0.05,
              base_seed=42):
    rng = np.random.default_rng(base_seed)
    theta = flat(warm_start(init_mlp(seed=1)))
    B = len(Js_train)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    hist = []
    for ep in range(1, epochs + 1):
        sig = sigma * (1.0 - 0.7 * ep / epochs)
        base = rng.normal(0.0, sig, (K // 2, len(theta)))
        eps_mat = np.concatenate([base, -base], axis=0)
        thetas = [theta + e for e in eps_mat]
        final = rollout_mom(Js_train, steps, thetas, seed=base_seed * 1000 + ep)
        rewards = final.reshape(K, B).T
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-9
        adv = (rewards - mean) / std
        grad = (adv @ eps_mat).mean(axis=0) / sig
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        theta -= lr * (m / (1 - b1 ** ep)) / (np.sqrt(v / (1 - b2 ** ep)) + eps_a)
        hist.append(rewards.mean())
        if ep % 10 == 0 or ep == epochs:
            print(f"  ep {ep:3d}: reward {rewards.mean():+.4f} "
                  f"(best {rewards.min():+.4f})", flush=True)
    return unflat(theta), np.array(hist)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    from cim import sk_instance
    from momentum import momentum_run
    from cim import energy

    N = 100
    Js_train = [sk_instance(N, s) for s in range(1, 17)]
    Js_val = [sk_instance(N, s) for s in range(101, 109)]

    # 基线: 动量动力学无温度 (momentum_run, γ=0.3, dt=0.05, 3000 步)
    base = []
    for J in Js_val:
        for sd in range(8):
            s, _, _ = momentum_run(J, steps=3000, dt=0.05, gamma=0.3, seed=400 + sd)
            base.append(energy(J, s))
    print(f"动量基线 (γ=0.3, 3000步): mean={np.mean(base):+.4f} best={np.min(base):+.4f}")

    t0 = time.time()
    params, hist = train_mom(Js_train, epochs=120, K=12)
    print(f"动量闭环训练完成 ({time.time()-t0:.0f}s)")
    np.save("theta_cl_mom.npy", flat(params))
    np.save("hist_cl_mom.npy", hist)

    # 评估: 学到的调度部署在动量动力学上 (600 步, 同种子)
    from closed_loop_v2 import rollout_cl_batch as _  # noqa
    def eval_mom(params, seeds=8):
        vals = []
        for sd in range(seeds):
            f = rollout_mom(list(Js_val), 600, [flat(params)], seed=777 + sd)
            vals.append(f)
        return np.mean(vals), np.std(vals), np.min(vals)
    m, s, mn = eval_mom(params)
    print(f"动量闭环策略 (600步): mean={m:+.4f} ± {s:.4f}, best={mn:+.4f}")
