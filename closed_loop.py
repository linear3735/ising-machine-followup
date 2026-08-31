"""
学习式退火调度 v3 —— 闭环反馈策略（用 Wong 的"有效能隙"可观测量驱动）。

设计（本仓库自己的设计）：
  状态 s_t = (近零自旋密度 mu, 当前最好能量 best_eps, 进度 t/T, log T)
           mu := frac(|x| < 0.01)   —— Wong 论文的 effective gap 指标
  动作:    T <- T * exp(0.2 * tanh(MLP(s_t)))，每 control_every 步做一次
  策略:    小 MLP (4 -> 16 -> 16 -> 1)，参数 theta
  训练:    ES + 实例内分组优势(GRPO式) + 共用噪声种子；a 固定 0
基线:     增益退火(a:0->0.5)、温度退火(开环线性, 王教授解析)
"""
import numpy as np
import time
from cim import sk_instance, energy, cim_run, \
    gain_anneal_schedule, const_schedule

DT = 0.02
T_MIN = 1e-4


def init_mlp(seed, d=4, h=16):
    r = np.random.default_rng(seed)
    params = {
        "W1": r.normal(0, 0.05, (d, h)), "b1": np.zeros(h),
        "W2": r.normal(0, 0.05, (h, h)), "b2": np.zeros(h),
        "W3": r.normal(0, 0.05, (h, 1)), "b3": np.zeros(1),
    }
    return params


def flat(params):
    return np.concatenate([params["W1"].ravel(), params["b1"],
                           params["W2"].ravel(), params["b2"],
                           params["W3"].ravel(), params["b3"]])


def unflat(v):
    i = 0
    d, h = 4, 16
    W1 = v[i:i + d * h].reshape(d, h); i += d * h
    b1 = v[i:i + h]; i += h
    W2 = v[i:i + h * h].reshape(h, h); i += h * h
    b2 = v[i:i + h]; i += h
    W3 = v[i:i + h * 1].reshape(h, 1); i += h
    b3 = v[i:i + 1]
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def mlp_apply(params, s):
    h = np.tanh(s @ params["W1"] + params["b1"])
    h = np.tanh(h @ params["W2"] + params["b2"])
    return h @ params["W3"] + params["b3"]


def rollout_cl(Js, steps, params, seed, control_every=20):
    """闭环 rollout (残差控制版): T(t) = T_analytic(t) * exp(0.3*tanh(MLP(s)))
    策略只输出对解析温度调度的有界修正。返回 sign、每实例最终能量、mu 轨迹"""
    B = len(Js)
    N = Js[0].shape[0]
    r = np.random.default_rng(seed)
    xs = r.normal(0.0, 0.01, (B, N))
    offset = np.zeros((B, 1))
    best_eps = np.zeros((B, 1))
    mu_traj = []
    for t in range(steps):
        frac = t / max(1, steps - 1)
        base_T = 1e-4 + 0.5 * (1.0 - frac)          # 王教授解析温度调度
        T = base_T * np.exp(offset) + T_MIN
        Jx = np.einsum('bij,bj->bi', Js, xs)
        xs = xs + DT * (-xs ** 3 - Jx)
        xs = xs + np.sqrt(2.0 * T * DT) * r.standard_normal((B, N))
        if t % control_every == 0:
            sgn = np.sign(xs)
            eps = (np.einsum('bi,bij,bj->b', sgn, Js, sgn) / (2.0 * N))[:, None]
            best_eps = np.minimum(best_eps, eps)
            mu = (np.abs(xs) < 0.01).mean(axis=1)[:, None]
            state = np.concatenate([mu, best_eps, np.full((B, 1), frac), offset], axis=1)
            offset = 0.3 * np.tanh(mlp_apply(params, state))
            mu_traj.append(float(mu.mean()))
    sgn = np.sign(xs)
    final = np.einsum('bi,bij,bj->b', sgn, Js, sgn) / (2.0 * N)
    return sgn, final, np.array(mu_traj)


def warm_start(params, iters=150, lr=0.02, seed=3, control_every=20, steps=1500):
    """行为克隆: 目标是'偏移=0'(即精确复现解析调度)。小权重初始化已近似 0,
    这里只需轻微校准让策略在各种状态输入下都输出 ~0。"""
    r = np.random.default_rng(seed)
    theta = flat(params)
    for it in range(iters):
        B = 128
        frac = r.random(B)
        s = np.stack([r.random(B) * 0.5, r.uniform(-0.75, 0.0, B),
                      frac, np.zeros(B)], axis=1)
        y = np.zeros((B, 1))
        p = unflat(theta)
        h1 = s @ p["W1"] + p["b1"]; a1 = np.tanh(h1)
        h2 = a1 @ p["W2"] + p["b2"]; a2 = np.tanh(h2)
        out = a2 @ p["W3"] + p["b3"]
        dout = 2.0 * (out - y) / B
        db3 = dout.sum(axis=0); dW3 = a2.T @ dout
        da2 = dout @ p["W3"].T; dh2 = da2 * (1 - a2 ** 2)
        db2 = dh2.sum(axis=0); dW2 = a1.T @ dh2
        da1 = dh2 @ p["W2"].T; dh1 = da1 * (1 - a1 ** 2)
        db1 = dh1.sum(axis=0); dW1 = s.T @ dh1
        theta -= lr * flat({"W1": dW1, "b1": db1, "W2": dW2,
                            "b2": db2, "W3": dW3, "b3": db3})
    return unflat(theta)


def train_cl(Js_train, steps=1500, epochs=150, K=10, sigma=0.1, lr=0.15,
             base_seed=42):
    rng = np.random.default_rng(base_seed)
    params0 = init_mlp(seed=1)
    params0 = warm_start(params0)
    theta = flat(params0)
    B = len(Js_train)
    hist = []
    for ep in range(epochs):
        rewards = np.zeros((B, K))
        sig = sigma * (1.0 - 0.7 * ep / epochs)          # σ 退火
        base = rng.normal(0.0, sig, (K // 2, len(theta)))
        eps_mat = np.concatenate([base, -base], axis=0)  # 对偶采样(方差缩减)
        seed_ep = base_seed * 1000 + ep
        for k in range(K):
            params_k = unflat(theta + eps_mat[k])
            _, final, _ = rollout_cl(Js_train, steps, params_k, seed=seed_ep)
            rewards[:, k] = final
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-9
        adv = (rewards - mean) / std
        grad = (adv @ eps_mat).mean(axis=0) / sig
        theta -= lr * grad
        hist.append(rewards.mean())
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d}: reward {rewards.mean():+.4f} "
                  f"(best {rewards.min():+.4f})", flush=True)
    return unflat(theta), np.array(hist)


def eval_cl(Js_val, params, seeds=8, steps=2000):
    vals = []
    for sd in range(seeds):
        _, final, _ = rollout_cl(list(Js_val), steps, params, seed=777 + sd)
        vals.append(final)
    vals = np.array(vals)
    return vals.mean(), vals.std(), vals.min()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    N = 100
    Js_train = [sk_instance(N, s) for s in range(1, 17)]
    Js_val = [sk_instance(N, s) for s in range(101, 109)]
    print(f"闭环训练(残差控制): {len(Js_train)} 实例 x K={12}, N={N}")
    t0 = time.time()
    params, hist = train_cl(Js_train, epochs=150, K=12)
    print(f"训练完成 ({time.time()-t0:.0f}s)")

    print("\n=== held-out 8 实例 x 8 种子, N=100 ===")
    # 开环基线
    gain_vals, ref_vals = [], []
    for J in Js_val:
        for sd in range(8):
            gs = gain_anneal_schedule(0.0, 0.5, 2000)
            s, _ = cim_run(J, 2000, gain_schedule=gs,
                           temp_schedule=const_schedule(1e-4), seed=200 + sd)
            gain_vals.append(energy(J, s))
    # 温度退火开环: T 0.5->1e-4 线性, a=0
    def temp_sched(t):
        return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / 2000)
    for J in Js_val:
        for sd in range(8):
            s, _ = cim_run(J, 2000, gain_schedule=const_schedule(0.0),
                           temp_schedule=temp_sched, seed=300 + sd)
            ref_vals.append(energy(J, s))
    m_g, s_g = np.mean(gain_vals), np.std(gain_vals)
    m_r, s_r = np.mean(ref_vals), np.std(ref_vals)
    m_l, s_l, mn_l = eval_cl(Js_val, params)
    print(f"  增益退火(传统)      : mean={m_g:+.4f} ± {s_g:.4f}")
    print(f"  温度退火(开环解析)   : mean={m_r:+.4f} ± {s_r:.4f}")
    print(f"  闭环学习策略(本文)   : mean={m_l:+.4f} ± {s_l:.4f}, best={mn_l:+.4f}")
    np.save("theta_cl.npy", flat(params))
    np.save("hist_cl.npy", hist)
