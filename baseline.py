"""Baseline: 复现 Wong 论文的增益退火 vs 温度退火对比 (SK 模型)。"""
import numpy as np
import time
from cim import (sk_instance, energy, sa_baseline, cim_run,
                 gain_anneal_schedule, const_schedule, temp_anneal_schedule)


def run_comparison(N=200, seeds=(1, 2), steps=3000, dt=0.02):
    rows = []
    for seed in seeds:
        J = sk_instance(N, seed)
        _, e_sa = sa_baseline(J, seed + 100)
        print(f"N={N} seed={seed}: SA 基态参考 eps={e_sa:.4f}", flush=True)

        configs = {
            # 增益退火（传统做法）: a 0->a_max, T 固定
            "gain_a0->0.5, T=1e-4": (gain_anneal_schedule(0.0, 0.5, steps), const_schedule(1e-4)),
            "gain_a0->1.0, T=1e-4": (gain_anneal_schedule(0.0, 1.0, steps), const_schedule(1e-4)),
            "gain_a0->0.5, T=0.03": (gain_anneal_schedule(0.0, 0.5, steps), const_schedule(0.03)),
            # 温度退火（论文主张的最优）: a 固定, T 高->低
            "temp_T0.1->1e-4, a=0.5": (const_schedule(0.5), temp_anneal_schedule(0.1, 1e-4, steps)),
            "temp_T0.5->1e-4, a=0.5": (const_schedule(0.5), temp_anneal_schedule(0.5, 1e-4, steps)),
            "temp_T0.5->1e-4, a=0.0": (const_schedule(0.0), temp_anneal_schedule(0.5, 1e-4, steps)),
            "temp_T1.0->1e-4, a=0.5": (const_schedule(0.5), temp_anneal_schedule(1.0, 1e-4, steps)),
        }
        for name, (gs, ts) in configs.items():
            t0 = time.time()
            s, rec = cim_run(J, steps, dt=dt, gain_schedule=gs, temp_schedule=ts, seed=7)
            e = energy(J, s)
            gap = e - e_sa
            rows.append((seed, name, e, gap, rec[:, 1].min()))
            print(f"  {name:28s}: eps={e:+.4f} (离SA {gap:+.4f}), 轨迹最低={rec[:,1].min():+.4f}, {time.time()-t0:.1f}s", flush=True)
    return rows


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print("=" * 70)
    run_comparison(N=200, seeds=(1,), steps=3000)
    print("=" * 70)
    run_comparison(N=100, seeds=(1,), steps=3000)
