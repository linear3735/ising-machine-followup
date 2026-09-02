"""closed_loop_v2.py —— 闭环训练提速版（与 v1 完全同构, 只改效率）

1. K 个 ES 扰动的 rollout 合并为一次批量 rollout (K*B, N) —— 省 K-1 次独立 rollout;
2. 控制动作(MLP)按 k 循环(便宜), 动力学全部批量化(matmul);
3. Adam 取代固定步长 SGD —— OpenAI-ES 标配, 收敛更快;
4. 其余与 closed_loop.py 相同: 残差控制 T(t)=T_analytic*exp(0.3*tanh(MLP)),
   实例内分组优势(z-score), 对偶采样, σ 退火, 共用噪声种子。
"""
import numpy as np
import time
from closed_loop import init_mlp, warm_start, flat, unflat, mlp_apply, DT, T_MIN


def rollout_cl_batch(Js, steps, thetas, seed, control_every=20):
    """thetas: list[flat 参数]; 一次批量 rollout (K*B, N)。返回 final (K*B,)"""
    K = len(thetas)
    B = len(Js)
    N = Js[0].shape[0]
    Jbig = np.repeat(np.stack(Js), K, axis=0)          # (K*B, N, N)
    r = np.random.default_rng(seed)
    xs = r.normal(0.0, 0.01, (K * B, N))
    offset = np.zeros((K * B, 1))
    best_eps = np.zeros((K * B, 1))
    for t in range(steps):
        frac = t / max(1, steps - 1)
        base_T = 1e-4 + 0.5 * (1.0 - frac)
        T = base_T * np.exp(offset) + T_MIN
        Jx = np.matmul(Jbig, xs[:, :, None])[:, :, 0]  # BLAS 批量 matmul
        xs = xs + DT * (-xs ** 3 - Jx)
        xs = xs + np.sqrt(2.0 * T * DT) * r.standard_normal((K * B, N))
        if t % control_every == 0:
            sgn = np.sign(xs)
            eps = np.matmul(sgn[:, None, :], np.matmul(Jbig, sgn[:, :, None]))[:, 0, 0] / (2.0 * N)
            best_eps = np.minimum(best_eps, eps[:, None])
            mu = (np.abs(xs) < 0.01).mean(axis=1)[:, None]
            # 控制: 每 k 一组参数, 对 (B,4) 状态批量算 MLP (便宜)
            for k in range(K):
                p = unflat(thetas[k])
                st = np.concatenate([mu[k * B:(k + 1) * B], best_eps[k * B:(k + 1) * B],
                                     np.full((B, 1), frac), offset[k * B:(k + 1) * B]], axis=1)
                offset[k * B:(k + 1) * B] = 0.3 * np.tanh(mlp_apply(p, st))
    sgn = np.sign(xs)
    final = np.matmul(sgn[:, None, :], np.matmul(Jbig, sgn[:, :, None]))[:, 0, 0] / (2.0 * N)
    return final


def train_cl_v2(Js_train, steps=1500, epochs=150, K=12, sigma=0.1,
                lr=0.05, base_seed=42):
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
        final = rollout_cl_batch(Js_train, steps, thetas, seed=base_seed * 1000 + ep)
        rewards = final.reshape(K, B).T                       # (B, K)
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-9
        adv = (rewards - mean) / std
        grad = (adv @ eps_mat).mean(axis=0) / sig
        # Adam
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mhat = m / (1 - b1 ** ep)
        vhat = v / (1 - b2 ** ep)
        theta -= lr * mhat / (np.sqrt(vhat) + eps_a)
        hist.append(rewards.mean())
        if ep % 10 == 0 or ep == epochs:
            print(f"  ep {ep:3d}: reward {rewards.mean():+.4f} "
                  f"(best {rewards.min():+.4f})", flush=True)
    return unflat(theta), np.array(hist)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    from cim import sk_instance
    N = 100
    Js_train = [sk_instance(N, s) for s in range(1, 17)]
    print(f"v2 闭环训练: {len(Js_train)} 实例 x K={12}, N={N} (批量 rollout + Adam)")
    t0 = time.time()
    params, hist = train_cl_v2(Js_train, epochs=150, K=12)
    print(f"训练完成 ({time.time()-t0:.0f}s)")
    np.save("theta_cl_v2.npy", flat(params))
    np.save("hist_cl_v2.npy", hist)
