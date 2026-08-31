"""digCIM + 闭环 μ 策略：T(t) = T_lin(t) * exp(0.3*tanh(MLP(μ, cut, f, offset)))
在 G1/G2/G3 上训练，reward = 终态 cut（对齐论文 TTS 的 success 口径）"""
import numpy as np
import time
from gset_bench import load_gset, cut
from closed_loop import init_mlp, flat, unflat, mlp_apply

DT = 0.03
A_GAIN = -10.0
T0 = 3.0


def digcim_closed(Js, steps, params, seed, control_every=50):
    """返回每实例终态 cut。状态: (μ, best_cut/1000, frac, offset)"""
    Js = np.stack(Js)
    B = len(Js)
    N = Js.shape[1]
    r = np.random.default_rng(seed)
    xs = r.normal(0, 0.01, (B, N))
    offset = np.zeros((B, 1))
    best = np.zeros((B, 1))
    for t in range(steps):
        f = t / max(1, steps - 1)
        T_lin = T0 * (1.0 - f)
        T = T_lin * np.exp(offset) + 1e-5
        xs = xs + DT * (A_GAIN * xs - np.einsum('bij,bj->bi', Js, np.sign(xs)))
        xs = xs + np.sqrt(2.0 * T * DT) * r.standard_normal((B, N))
        xs = np.clip(xs, -1.0, 1.0)
        if t % control_every == 0:
            sgn = np.sign(xs)
            c = (Js.sum(axis=(1, 2)) / 4.0 -
                 np.einsum('bi,bij,bj->b', sgn, Js, sgn) / 4.0)[:, None]
            best = np.maximum(best, c)
            mu = (np.abs(xs) < 0.01).mean(axis=1)[:, None]
            state = np.concatenate([mu, best / 1000.0,
                                    np.full((B, 1), f), offset], axis=1)
            offset = 0.5 * np.tanh(mlp_apply(params, state))
    sgn = np.sign(xs)
    final = (Js.sum(axis=(1, 2)) / 4.0 -
             np.einsum('bi,bij,bj->b', sgn, Js, sgn) / 4.0)
    return final


def train(Js, steps=3000, epochs=60, K=8, sigma=0.1, lr=0.2, base_seed=7):
    rng = np.random.default_rng(base_seed)
    theta = flat(init_mlp(seed=2))
    B = len(Js)
    for ep in range(epochs):
        sig = sigma * (1.0 - 0.7 * ep / epochs)
        base = rng.normal(0, sig, (K // 2, len(theta)))
        eps_mat = np.concatenate([base, -base], axis=0)
        rewards = np.zeros((B, K))
        seed_ep = base_seed * 1000 + ep
        for k in range(K):
            params_k = unflat(theta + eps_mat[k])
            rewards[:, k] = digcim_closed(Js, steps, params_k, seed=seed_ep)
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-9
        adv = (rewards - mean) / std
        theta -= lr * (adv @ eps_mat).mean(axis=0) / sig
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep {ep:2d}: reward {rewards.mean():.1f} "
                  f"(best {rewards.max():.0f})", flush=True)
    return unflat(theta)


if __name__ == "__main__":
    Js = [load_gset(f"{g}.dat") for g in ("G1", "G2", "G3")]
    print(f"训练 digCIM 闭环策略 ({len(Js)} 实例, K=8, steps=3000)")
    t0 = time.time()
    params = train(Js)
    print(f"训练完成 ({time.time()-t0:.0f}s)")
    np.save("digcim_theta_cl.npy", flat(params))

    # 对比: 基线(线性调度) vs 闭环, 指标 = 终态 cut 与 沿途 best
    known = {"G1": 11624, "G2": 11620, "G3": 11622}
    for gi, g in enumerate(("G1", "G2", "G3")):
        J = Js[gi:gi + 1]
        base_fin, cl_fin = [], []
        for sd in range(16):
            base_fin.append(digcim_closed(J, 3000, unflat(flat(init_mlp(seed=99)) * 0), seed=400 + sd)[0])
            cl_fin.append(digcim_closed(J, 3000, params, seed=400 + sd)[0])
        base_fin, cl_fin = np.array(base_fin), np.array(cl_fin)
        print(f"{g}: 线性调度 终态 mean={base_fin.mean():.1f} best={base_fin.max():.0f} | "
              f"闭环 终态 mean={cl_fin.mean():.1f} best={cl_fin.max():.0f} "
              f"(已知最优 {known[g]})")
