"""clean_eval_cl.py —— 闭环策略干净重评 (训练 1-16, 评估 101-108, 无泄漏)

协议: N∈{100,200}, 实例 101-108 (N=200 用同种子生成新 J), 16 种子,
2000 步 dt=0.02 (40tu), 闭环(θ_cl) vs 温度退火(解析线性) vs 增益退火。
判据: pooled mean±std, best, 每实例 16 种子 std 的平均 (方差声称), 配对 Δ。
"""
import json
import numpy as np
from cim import sk_instance, cim_run, gain_anneal_schedule, const_schedule
from closed_loop import rollout_cl, unflat

STEPS = 2000
SEEDS = 16


def temp_sched(t):
    return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / STEPS)


def main():
    params = unflat(np.load("theta_cl.npy"))
    out = {}
    for N in (100, 200):
        Js_val = [sk_instance(N, s) for s in range(101, 109)]   # 8 held-out
        # 闭环: 每种子一次批量 rollout (8 实例共享噪声)
        cl = np.stack([rollout_cl(Js_val, STEPS, params, seed=777 + sd)[1]
                       for sd in range(SEEDS)])                 # (16, 8)
        # 基线: 每 (实例, 种子)
        temp, gain = [], []
        for J in Js_val:
            tv, gv = [], []
            for sd in range(SEEDS):
                s, _ = cim_run(J, STEPS, gain_schedule=const_schedule(0.0),
                               temp_schedule=temp_sched, seed=300 + sd)
                tv.append(energy(J, s))
                s, _ = cim_run(J, STEPS,
                               gain_schedule=gain_anneal_schedule(0.0, 0.5, STEPS),
                               temp_schedule=const_schedule(1e-4), seed=200 + sd)
                gv.append(energy(J, s))
            temp.append(tv)
            gain.append(gv)
        temp = np.array(temp).T      # (16, 8)
        gain = np.array(gain).T
        # 统计
        res = {}
        for name, v in (("closed", cl), ("temp", temp), ("gain", gain)):
            per_inst_std = v.std(axis=0)          # 每实例 16 种子 std
            res[name] = dict(mean=float(v.mean()), std=float(v.std()),
                             best=float(v.min()),
                             mean_inst_std=float(per_inst_std.mean()),
                             per_inst_mean=float(v.mean(axis=0).mean()))
        # 配对 (同实例同种子序): closed - temp
        d = cl - temp
        res["paired"] = dict(d_mean=float(d.mean()), d_std=float(d.std()),
                             frac_better=float((d < 0).mean()),
                             n=float(d.size))
        out[N] = res
        print(f"\n=== N={N}, 实例 101-108, {SEEDS} 种子, 2000 步 (40tu) ===")
        for name in ("closed", "temp", "gain"):
            r = res[name]
            print(f"  {name:6s}: mean={r['mean']:+.4f} ± {r['std']:.4f} "
                  f"(每实例std均值 {r['mean_inst_std']:.4f}) best={r['best']:+.4f}")
        p = res["paired"]
        print(f"  配对 closed-temp: Δ={p['d_mean']:+.4f} ± {p['d_std']:.4f}, "
              f"closed 更优占比 {p['frac_better']:.0%}")
    json.dump(out, open("clean_eval_cl.json", "w"), indent=2)
    print("\nsaved clean_eval_cl.json")


def energy(J, s):
    return (s @ J @ s) / (2.0 * J.shape[0])


if __name__ == "__main__":
    main()
