"""
学习式退火调度 v2：在王教授解析温度退火基础上做数据驱动改进。

调度族:  T(t) = T_hi  (t < sw*T);  T_hi*((1-t/T)/(1-sw))^p  (t >= sw*T)
         a(t) = a (常数);  参数 theta = (log T_hi, log p, a, sw_logit)
参考 (王教授解析/最优线性基线): theta_ref = (log 0.5, log 1, 0, -6)

方法:
  - ES 风格 score-function 梯度: 每个实例×K 条 rollout, theta+sigma*eps
  - 方差缩减: 同 epoch 内 K 条 rollout 共用同一噪声种子(common random numbers)
  - GRPO 式实例内分组优势: adv = (r - 组均值)/(组标准差)
  - L2 罚项拉向 theta_ref (连续参数版 KL-to-reference)
"""
import numpy as np
import time
from cim import sk_instance, energy, sa_baseline, cim_run, \
    gain_anneal_schedule, const_schedule

DT = 0.02
T_MIN = 1e-4
N_PARAMS = 4


def decode(theta):
    """raw -> (T0, p, a, sw) 每列"""
    T0 = np.clip(np.exp(theta[..., 0]), 0.05, 3.0)
    p = np.clip(np.exp(theta[..., 1]), 0.1, 6.0)
    a = np.clip(theta[..., 2], -1.0, 1.0)
    sw = 1.0 / (1.0 + np.exp(-theta[..., 3])) * 0.8     # 平台占比 ∈ (0, 0.8)
    return T0, p, a, sw


def rollout_batch(Js, steps, thetas, seed):
    """批量 rollout：thetas (B,4)。返回 signs (B,N)。"""
    B = len(Js)
    N = Js[0].shape[0]
    T0, p, a, sw = decode(thetas)
    T0 = T0[:, None]; p = p[:, None]; a = a[:, None]; sw = sw[:, None]
    r = np.random.default_rng(seed)
    xs = r.normal(0.0, 0.01, (B, N))
    Jx = np.einsum('bij,bj->bi', Js, xs)
    for t in range(steps):
        frac = t / max(1, steps - 1)
        cool = np.clip((1.0 - frac) / np.maximum(1e-6, 1.0 - sw), 0.0, 1.0) ** p
        T = np.where(frac < sw, T0, T0 * cool) + T_MIN
        xs = xs + DT * (-xs ** 3 + a * xs - Jx)
        xs = xs + np.sqrt(2.0 * T * DT) * r.standard_normal((B, N))
        Jx = np.einsum('bij,bj->bi', Js, xs)
    return np.sign(xs)


def final_energy(Js, signs):
    N = Js[0].shape[0]
    return np.einsum('bi,bij,bj->b', signs, Js, signs) / (2.0 * N)


def train(Js_train, theta_ref, sigma=0.12, lr=0.10, lam=0.02,
          epochs=120, K=8, steps=1500, base_seed=42):
    rng = np.random.default_rng(base_seed)
    B = len(Js_train)
    theta = np.array(theta_ref, dtype=float).copy()
    theta_ref_np = np.array(theta_ref, dtype=float)
    hist_r, hist_t = [], []
    for ep in range(epochs):
        eps_mat = rng.normal(0.0, sigma, (B, K, N_PARAMS))      # θ 扰动
        rewards = np.zeros((B, K))
        seed_ep = base_seed * 1000 + ep
        for k in range(K):                                      # 共用噪声种子(方差缩减)
            thetas = theta[None, :] + eps_mat[:, k, :]
            signs = rollout_batch(Js_train, steps, thetas, seed=seed_ep)
            rewards[:, k] = final_energy(Js_train, signs)
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-9
        adv = (rewards - mean) / std
        grad = (adv[..., None] * eps_mat).mean(axis=(0, 1)) / sigma
        grad += lam * (theta - theta_ref_np)
        theta -= lr * grad
        hist_r.append(rewards.mean())
        hist_t.append(theta.copy())
        if ep % 10 == 0 or ep == epochs - 1:
            T0, p, a, sw = decode(theta)
            print(f"  ep {ep:3d}: reward {rewards.mean():+.4f} | "
                  f"T_hi={T0:.3f} p={p:.2f} a={a:+.3f} sw={sw:.2f}", flush=True)
    return theta, np.array(hist_r), np.array(hist_t)


def eval_schedule(Js_val, theta, seeds=8, steps=2000):
    vals = []
    for sd in range(seeds):
        B = len(Js_val)
        thetas = np.tile(np.array(theta), (B, 1))
        signs = rollout_batch(list(Js_val), steps, thetas, seed=777 + sd)
        vals.append(final_energy(list(Js_val), signs))
    vals = np.array(vals)
    return vals.mean(), vals.std(), vals.min()


def eval_gain_anneal(Js_val, seeds=8, steps=2000):
    """真正的增益退火: a 从 0 升到 0.5, T=1e-4 固定"""
    vals = []
    for J in Js_val:
        for sd in range(seeds):
            gs = gain_anneal_schedule(0.0, 0.5, steps)
            s, _ = cim_run(J, steps, gain_schedule=gs,
                           temp_schedule=const_schedule(1e-4), seed=200 + sd)
            vals.append(energy(J, s))
    vals = np.array(vals)
    return vals.mean(), vals.std(), vals.min()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    N = 100
    Js_train = [sk_instance(N, s) for s in range(1, 13)]
    Js_val = [sk_instance(N, s) for s in range(101, 109)]
    theta_ref = (np.log(0.5), np.log(1.0), 0.0, -6.0)   # 线性温度退火, a=0

    print(f"训练: {len(Js_train)} 实例 x K={8}, N={N}, epochs=120")
    t0 = time.time()
    theta_star, hist_r, hist_t = train(Js_train, theta_ref)
    print(f"训练完成 ({time.time()-t0:.0f}s), 参数历史:")
    T0, p, a, sw = decode(theta_star)
    print(f"  学习到: T_hi={T0:.3f}, p={p:.2f}, a={a:+.3f}, sw={sw:.2f}")

    print("\n=== held-out 8 实例 x 8 种子, N=100 ===")
    for name, th in [("增益退火(传统)", None),
                     ("温度退火(王教授解析)", theta_ref),
                     ("学习调度(本文)", theta_star)]:
        if th is None:
            m, sd, mn = eval_gain_anneal(Js_val)
        else:
            m, sd, mn = eval_schedule(Js_val, th)
        print(f"  {name:22s}: mean={m:+.4f} ± {sd:.4f}, best={mn:+.4f}")
    np.save("theta_star.npy", np.array(theta_star))
    np.save("hist_r.npy", hist_r)
