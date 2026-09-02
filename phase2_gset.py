"""Phase 2: smomentum_digcim 扩 Gset —— G11-G21（800 节点稀疏族）+ G22-G26（2000 节点大实例）

协议（与论文同口径、诚实）:
  - 成功定义 = 终态 cut 达到已知最优（与论文 SM 的 success/TTS 定义一致）;
  - digCIM 基线: 论文配方 (a=-10, dt=0.03, T0=3, clip ±1, 5000 步), 终态;
  - smomentum: 阻尼辛欧拉 × digCIM 驱动 × 温度退火 × 墙,
    2000 步 (论文 5000 步的 40%), 扫 γ∈{3,5} × dt∈{0.2,0.25,0.3};
  - 两种方法同种子; 同时报告 running-best 与终态双口径, 劣解率(<0.9×known) 作发散指示;
  - TTS_steps = steps·ln(0.01)/ln(1−Ps)。

用法: python phase2_gset.py [--instances G11,G12,...] [--seeds 16]
"""
import argparse
import json
import time

import numpy as np

from gset_bench import load_gset, cut, digcim_run
from momentum import smomentum_digcim

# Gset 已知最优 (标准 best-known 表; 交叉核对: p-bit SA 论文 2601.15561 Tab.1,
# Adam-for-AIM 论文 2606.03917 复制的标准表; G16=3052, G17=3047 已修正)
KNOWN = {
    "G11": 564, "G12": 556, "G13": 582, "G14": 3064, "G15": 3050,
    "G16": 3052, "G17": 3047, "G18": 992, "G19": 906, "G20": 941, "G21": 931,
    "G22": 13359, "G23": 13344, "G24": 13337, "G25": 13340, "G26": 13328,
}


def smomentum_stats(A, steps, dt, gamma, a, Tinit, seeds):
    """返回 (final_cuts, run_best_cuts) 数组; 终态与全程 best 双口径"""
    fin, rb = [], []
    for sd in seeds:
        N = A.shape[0]
        r = np.random.default_rng(sd)
        x = r.normal(0, 0.01, N)
        y = r.normal(0, 0.01, N)
        best = -1e18
        for t in range(steps):
            T = Tinit * (1.0 - t / max(1, steps - 1))
            F = a * x - (A @ np.sign(x))
            y = (y + dt * F) / (1.0 + dt * gamma)
            y = y + np.sqrt(2.0 * T * dt) * r.standard_normal(N)
            x = x + dt * y
            mask = np.abs(x) > 1.0
            x[mask] = np.sign(x[mask])
            y[mask] = 0.0
            if t % 100 == 0 or t == steps - 1:
                best = max(best, cut(A, np.sign(x)))
        fin.append(cut(A, np.sign(x)))
        rb.append(best)
    return np.array(fin), np.array(rb)


def summarize(name, cuts, steps, known):
    cuts = np.asarray(cuts)
    ps = float((cuts >= known).mean())
    bad = float((cuts < 0.9 * known).mean())
    tts = float(steps * np.log(0.01) / np.log(1 - ps)) if ps > 0 else None
    return dict(name=name, steps=steps, best=int(cuts.max()), mean=round(float(cuts.mean()), 1),
                ps=round(ps, 4), bad=round(bad, 4),
                tts_steps=int(tts) if tts else None)


def run_instance(g, A, seeds, known, sm_grid):
    out = {"n": A.shape[0], "edges": int((A != 0).sum() / 2), "known": known}
    # digCIM 论文配方: 终态口径 (论文 success 定义) + running-best 参考
    fin, rb = [], []
    for sd in seeds:
        s, b = digcim_run(A, 5000, 0.03, -10.0, 3.0, sd)
        fin.append(cut(A, s))
        rb.append(b)
    out["digCIM_final"] = summarize("digCIM(论文配方5000步)", fin, 5000, known)
    out["digCIM_best"] = summarize("digCIM(running-best)", rb, 5000, known)
    # smomentum 扫参
    out["smomentum"] = {}
    for gamma, dt in sm_grid:
        fin, rb = smomentum_stats(A, 2000, dt, gamma, -10.0, 3.0, seeds)
        out["smomentum"][f"g{gamma}_dt{dt}"] = {
            "final": summarize(f"sm γ={gamma} dt={dt} (终态)", fin, 2000, known),
            "best": summarize(f"sm γ={gamma} dt={dt} (running-best)", rb, 2000, known),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=str,
                    default="G11,G12,G13,G14,G15,G16,G17,G18,G19,G20,G21")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--gamma", type=str, default="3,5")
    ap.add_argument("--dt", type=str, default="0.2,0.25,0.3")
    args = ap.parse_args()

    gnames = args.instances.split(",")
    seeds = list(range(1000, 1000 + args.seeds))
    sm_grid = [(float(g), float(d)) for g in args.gamma.split(",")
               for d in args.dt.split(",")]
    print(f"实例 {gnames} | 种子 {args.seeds} | smomentum 网格 {sm_grid}\n")

    results, t0 = {}, time.time()
    for g in gnames:
        A = load_gset(f"{g}.dat")
        t1 = time.time()
        r = run_instance(g, A, seeds, KNOWN[g], sm_grid)
        r["耗时s"] = round(time.time() - t1, 1)
        results[g] = r
        d = r["digCIM_final"]
        print(f"=== {g} (N={r['n']}, edges={r['edges']}, known={r['known']}) "
              f"[{r['耗时s']}s]")
        print(f"  digCIM 5000步: best={d['best']} mean={d['mean']} "
              f"Ps={d['ps']:.1%} TTS={d['tts_steps']}")
        for key, v in r["smomentum"].items():
            f = v["final"]
            print(f"  sm {key}: best={f['best']} mean={f['mean']} "
                  f"Ps={f['ps']:.1%} 劣解率={f['bad']:.0%} TTS={f['tts_steps']}")
        print()

    json.dump(results, open("phase2_gset_results.json", "w"), indent=2)
    print(f"总耗时 {time.time()-t0:.0f}s -> phase2_gset_results.json")
