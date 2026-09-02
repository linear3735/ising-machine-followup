"""closed_loop_restart.py —— v3: 闭环控制器 + 草案头(重启头), MTP 式共享表示

机制:
  state (mu, best_eps, frac, offset, t_since_best) -> MLP(5-16-16) -> 双头
    头0: 调度修正 (原残差规则, 泄漏积分式)
    头1: 重启强度 g (草案头; ES 可训离散决策, 无需可微)
  触发: g>0.1 且 t_since_best>=6 窗口 且 cooldown>=12 且 frac<0.9
  动作: x <- 0.9*sigma_best + 0.2*N(0,1);  offset <- 2.0*clip(g,0,1)   (再热)
  奖励: 全程 running best eps (有重启后终态无意义)
对照: none(无重启) / fixed(每30窗口) / learned(学习头)
"""
import numpy as np
import time
from cim import energy  # noqa

DT, T_MIN = 0.02, 1e-4


def init_mlp2(seed=1):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 0.05, (5, 16)), "b1": np.zeros(16),
            "W2": r.normal(0, 0.05, (16, 16)), "b2": np.zeros(16),
            "W3": r.normal(0, 0.05, (16, 2)), "b3": np.zeros(2)}


def flat2(p):
    return np.concatenate([p["W1"].ravel(), p["b1"], p["W2"].ravel(),
                           p["b2"], p["W3"].ravel(), p["b3"]])


def unflat2(v):
    i = 0
    W1 = v[i:i + 80].reshape(5, 16); i += 80
    b1 = v[i:i + 16]; i += 16
    W2 = v[i:i + 256].reshape(16, 16); i += 256
    b2 = v[i:i + 16]; i += 16
    W3 = v[i:i + 32].reshape(16, 2); i += 32
    b3 = v[i:i + 2]
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def mlp2(p, s):
    h = np.tanh(s @ p["W1"] + p["b1"])
    h = np.tanh(h @ p["W2"] + p["b2"])
    return h @ p["W3"] + p["b3"]


def rollout_clr(Js, steps, thetas, seed, variant="learned",
                control_every=20, restart_interval=30):
    """双头闭环 rollout。返回 (best_over_traj, final_eps, 重启次数)"""
    K = len(thetas)
    B = len(Js)
    N = Js[0].shape[0]
    Jbig = np.repeat(np.stack(Js), K, axis=0)
    r = np.random.default_rng(seed)
    xs = r.normal(0.0, 0.01, (K * B, N))
    offset = np.zeros((K * B, 1))
    best_eps = np.full((K * B, 1), np.inf)
    best_sgn = np.zeros((K * B, N))
    t_since = np.zeros((K * B, 1), int)
    cooldown = np.full((K * B, 1), 99, int)
    n_restart = 0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        base_T = 1e-4 + 0.5 * (1.0 - frac)
        T = base_T * np.exp(np.clip(offset, -3.0, 2.5)) + T_MIN
        Jx = np.matmul(Jbig, xs[:, :, None])[:, :, 0]
        xs = xs + DT * (-xs ** 3 - Jx)
        xs = xs + np.sqrt(2.0 * T * DT) * r.standard_normal((K * B, N))
        if t % control_every == 0:
            sgn = np.sign(xs)
            eps = np.matmul(sgn[:, None, :], np.matmul(Jbig, sgn[:, :, None]))[:, 0, 0] / (2.0 * N)
            improved = eps < best_eps[:, 0]
            best_eps = np.minimum(best_eps, eps[:, None])
            best_sgn[improved] = sgn[improved]
            t_since = np.where(improved[:, None], 0, t_since + 1)
            mu = (np.abs(xs) < 0.01).mean(axis=1)[:, None]
            for k in range(K):
                p = unflat2(thetas[k])
                sl = slice(k * B, (k + 1) * B)
                st = np.concatenate([mu[sl], best_eps[sl], np.full((B, 1), frac),
                                     offset[sl], (t_since[sl] / 75.0)], axis=1)
                out = mlp2(p, st)                                    # (B, 2)
                off = offset[sl] * 0.97 + 0.3 * np.tanh(out[:, 0:1])  # 泄漏积分调度
                g = out[:, 1:2]
                if variant == "learned":
                    trig = ((g > 0.1) & (t_since[sl] >= 6)
                            & (cooldown[sl] >= 12) & (frac < 0.9))[:, 0]
                    boost = 2.0 * np.clip(g[:, 0], 0, 1)
                elif variant == "fixed":
                    trig = ((cooldown[sl] >= restart_interval)
                            & (t_since[sl] >= 2) & (frac < 0.9))[:, 0]
                    boost = np.full(B, 2.0)
                else:
                    trig = np.zeros(B, bool)
                    boost = np.zeros(B)
                if trig.any():
                    idx = np.where(trig)[0]
                    xs[sl][idx] = (0.9 * best_sgn[sl][idx]
                                   + 0.2 * r.standard_normal((len(idx), N)))
                    off[idx] = boost[idx, None]
                    cooldown[sl][idx] = 0
                    t_since[sl][idx] = 0
                    n_restart += int(trig.sum())
                offset[sl] = off
            cooldown += 1
    sgn = np.sign(xs)
    final = np.matmul(sgn[:, None, :], np.matmul(Jbig, sgn[:, :, None]))[:, 0, 0] / (2.0 * N)
    return best_eps[:, 0], final, n_restart


def train_clr(Js_train, steps=1500, epochs=60, K=12, sigma=0.1, lr=0.05,
              variant="learned", base_seed=42):
    rng = np.random.default_rng(base_seed)
    theta = flat2(init_mlp2())
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
        best, final, nr = rollout_clr(Js_train, steps, thetas,
                                      seed=base_seed * 1000 + ep, variant=variant)
        rewards = best.reshape(K, B).T
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
                  f"(best {rewards.min():+.4f}, 重启 {nr})", flush=True)
    return unflat2(theta), np.array(hist)


def eval_clr(Js_val, theta, seeds=8, steps=1500, variant="learned"):
    bs, fs = [], []
    for sd in range(seeds):
        best, final, nr = rollout_clr(list(Js_val), steps, [flat2(theta)],
                                      seed=777 + sd, variant=variant)
        bs.append(best[0])
        fs.append(final[0])
    return np.mean(bs), np.std(bs), np.min(bs), np.mean(fs)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    from cim import sk_instance
    N = 100
    Js_train = [sk_instance(N, s) for s in range(1, 17)]
    Js_val = [sk_instance(N, s) for s in range(101, 109)]
    print(f"v3 草案头实验: N={N}, 训练 16 实例 x K=12 x {60} epochs, 奖励=running best")
    for variant in ("none", "fixed", "learned"):
        t0 = time.time()
        th, hist = train_clr(Js_train, variant=variant)
        m, s, mn, mf = eval_clr(Js_val, th, variant=variant)
        print(f"[{variant:>7}] 训练 {time.time()-t0:.0f}s | held-out "
              f"best-eps mean {m:+.4f} ± {s:.4f} (min {mn:+.4f}) | "
              f"final mean {mf:+.4f}")
