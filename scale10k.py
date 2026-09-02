"""scale10k.py —— 直接冲 N=10,000 (SK, 论文规模)

- J float32 (400MB), 8 种子批量 (N,S) matmul (13ms/步实测)
- 梯度温度退火 (a=0, T: 0.5->1e-4, dt=0.02) vs 动量 (γ=0.3, dt=0.05, tanh)
- 多阈值 TTS (-0.700/-0.720/-0.730/-0.735) + best/mean @ 8000 步
- 固定超参 (与 N=100-1000 补测一致, 无偷看)
"""
import numpy as np
import time
import json

CAP = 8000
THRS = (-0.700, -0.720, -0.730, -0.735)


def sk_f32(N, seed):
    r = np.random.default_rng(seed)
    J = r.normal(0.0, 1.0 / np.sqrt(N), (N, N)).astype(np.float32)
    J = np.triu(J, 1)
    J = J + J.T
    return J


def eps_batch(J, sgn):
    """sgn (N,S) -> E (S,) = s^T J s / 2N"""
    return (sgn * (J @ sgn)).sum(axis=0) / (2.0 * J.shape[0])


def tts_cross(rec_e, rec_t, thr):
    for e, t in zip(rec_e, rec_t):
        if e <= thr:
            return t
    return np.inf


def run_method(J, seeds, kind):
    """kind: 'grad' (梯度温度退火) 或 'mom' (动量 γ=0.3,dt=0.05)"""
    N = J.shape[0]
    S = len(seeds)
    rngs = [np.random.default_rng(sd) for sd in seeds]
    x = np.array([r.normal(0, 0.01, N) for r in rngs], dtype=np.float32).T  # (N,S)
    dt = 0.02 if kind == "grad" else 0.05
    y = None
    if kind == "mom":
        y = np.array([r.normal(0, 0.01, N) for r in rngs], dtype=np.float32).T
        dt_g = 1.0 / (1.0 + dt * 0.3)
    rec_t, rec_e = [], []
    best = np.full(S, 1e9, dtype=np.float64)
    T_anneal = CAP // 2
    for t in range(1, CAP + 1):
        if kind == "grad":
            T = max(1e-4, 0.5 * (1.0 - t / CAP))
            x = x + dt * (-x ** 3 - (J @ x))
            x = x + np.sqrt(2.0 * T * dt) * np.array(
                [r.standard_normal(N) for r in rngs], dtype=np.float32).T
        else:
            a = min(1.0, t / T_anneal)
            T = 1e-4
            y = (y + dt * (-(1.0 - a) * x - (J @ x))) * dt_g
            y = y + np.sqrt(2.0 * T * dt) * np.array(
                [r.standard_normal(N) for r in rngs], dtype=np.float32).T
            x = x + dt * y
            x = np.tanh(x)
        if t % 50 == 0 or t == CAP:
            e = eps_batch(J, np.sign(x))
            rec_t.append(t)
            rec_e.append(e)
            best = np.minimum(best, e)
    return rec_t, np.array(rec_e), best


if __name__ == "__main__":
    N = 10000
    seeds = list(range(0, 8))
    J = sk_f32(N, 101)
    print(f"J 生成完毕 (float32 {J.nbytes/1e6:.0f}MB), N={N}, 种子 {len(seeds)}", flush=True)
    out = {}
    for kind, label in (("grad", "梯度温度退火"), ("mom", "动量 γ=0.3")):
        t0 = time.time()
        rec_t, rec_e, best = run_method(J, seeds, kind)
        sec = time.time() - t0
        row = {"sec": round(sec)}
        print(f"\n=== {label} ({sec:.0f}s) ===")
        for thr in THRS:
            tt = np.array([tts_cross(rec_e[:, s], rec_t, thr) for s in range(len(seeds))])
            med = np.median(tt[np.isfinite(tt)]) if np.isfinite(tt).any() else np.inf
            row[f"tts_{thr}"] = med
            print(f"  TTS 到 {thr}: 中位 {med:.0f} 步 (达标 {np.isfinite(tt).mean():.0%})")
        row["best"] = float(best.min())
        row["mean"] = float(best.mean())
        print(f"  best {best.min():+.4f} mean {best.mean():+.4f} (8 种子)")
        out[kind] = row
    json.dump(out, open("scale10k.json", "w"), indent=2)
    print("\nsaved scale10k.json")
