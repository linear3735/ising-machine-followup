"""clean_eval_cl_v2.py —— v2 重训 + 干净重评 (训练 1-16, 评估 101-108)"""
import numpy as np
import time
from cim import sk_instance, cim_run, gain_anneal_schedule, const_schedule
from closed_loop_v2 import train_cl_v2
from closed_loop import rollout_cl, flat

STEPS = 2000
SEEDS = 16


def temp_sched(t):
    return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / STEPS)


def energy(J, s):
    return (s @ J @ s) / (2.0 * J.shape[0])


def eval_N(N, params):
    Js_val = [sk_instance(N, s) for s in range(101, 109)]
    cl = np.stack([rollout_cl(Js_val, STEPS, params, seed=777 + sd)[1]
                   for sd in range(SEEDS)])
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
    temp = np.array(temp).T
    gain = np.array(gain).T
    print(f"\n=== N={N} (v2 重训策略, 101-108 x {SEEDS} 种子, 2000 步 40tu) ===")
    for name, v in (("closed", cl), ("temp", temp), ("gain", gain)):
        print(f"  {name:6s}: mean={v.mean():+.4f} ± {v.std():.4f} "
              f"(每实例std均值 {v.std(axis=0).mean():.4f}) best={v.min():+.4f}")
    d = cl - temp
    print(f"  配对 closed-temp: Δ={d.mean():+.4f} ± {d.std():.4f}, "
          f"closed 更优 {100 * (d < 0).mean():.0f}%")
    return dict(closed=cl, temp=temp, gain=gain)


if __name__ == "__main__":
    t0 = time.time()
    Js_tr = [sk_instance(100, s) for s in range(1, 17)]
    params, hist = train_cl_v2(Js_tr, epochs=150, K=12)
    np.save("theta_cl_v2.npy", flat(params))
    print(f"训练完成 {time.time()-t0:.0f}s")
    r100 = eval_N(100, params)
    r200 = eval_N(200, params)
    np.savez("clean_eval_cl_v2.npz", **{f"{k}_{n}": v for n, d in
                                        (("100", r100), ("200", r200))
                                        for k, v in d.items()})
    print(f"\n总耗时 {time.time()-t0:.0f}s -> theta_cl_v2.npy, clean_eval_cl_v2.npz")
