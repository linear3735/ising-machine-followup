"""honest_rerun.py —— 诚实重述四方法对比 (修 provenance 漏洞)

协议 (全部先定后跑, 无偷看):
  1. dev 实例 seed=101 (N=100, 不在 eval 集): 扫 gamma ∈ {0,0.1,0.3,0.5,1.0},
     dt=0.05 固定 (SB 家族文档惯例), 8 种子 3000 步, 按 mean 能量选 gamma*;
  2. eval: N=100/200, 实例 102-105 (held-out), 8 种子, 等物理时间 60 tu
     (梯度 dt=0.02 x 3000 步 = 60tu; 动量 dt=0.05 x 1200 步 = 60tu; 增益同梯度);
  3. 阈值网格 N=100: {-0.68,-0.70,-0.72}; N=200: {-0.70,-0.72,-0.74};
     headline 用两方法达标率都>=50% 的最深阈值; 同时给步数(x dt=物理时间)与达标率。
"""
import json
import numpy as np
from cim import sk_instance, cim_run, gain_anneal_schedule, const_schedule
from momentum import momentum_run

DT_G = 0.02
DT_M = 0.05
TU = 60.0                       # 等物理时间预算
STEPS_G = int(TU / DT_G)        # 3000
STEPS_M = int(TU / DT_M)        # 1200


def temp_sched(t, cap):
    return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / cap)


def traj_grad(J, mode, cap, seed):
    if mode == "gain":
        gs = gain_anneal_schedule(0.0, 0.5, cap)
        _, rec = cim_run(J, cap, gain_schedule=gs,
                         temp_schedule=const_schedule(1e-4), seed=seed)
    else:
        _, rec = cim_run(J, cap, gain_schedule=const_schedule(0.0),
                         temp_schedule=lambda t: temp_sched(t, cap), seed=seed)
    return rec


def traj_mom(J, gamma, cap, seed):
    _, rec, _ = momentum_run(J, cap, dt=DT_M, gamma=gamma, seed=seed)
    return rec


def tts_stats(rec, thr):
    """rec: (t, e) 数组; 返回首次 <= thr 的 t (None=未达)"""
    idx = np.where(rec[:, 1] <= thr)[0]
    return int(rec[idx[0], 0]) if len(idx) else None


def run_instance(J, gamma, thr_list, seeds):
    out = {}
    for m, cap, dt in (("gain", STEPS_G, DT_G), ("temp", STEPS_G, DT_G)):
        out[m] = {}
        for sd in seeds:
            rec = traj_grad(J, m, cap, sd)
            for thr in thr_list:
                out[m].setdefault(thr, []).append(tts_stats(rec, thr))
    out["momentum"] = {}
    for sd in seeds:
        rec = traj_mom(J, gamma, STEPS_M, sd)
        for thr in thr_list:
            out["momentum"].setdefault(thr, []).append(tts_stats(rec, thr))
    # 终态质量 (60tu 末)
    fin = {}
    for m, cap, dt in (("gain", STEPS_G, DT_G), ("temp", STEPS_G, DT_G)):
        v = []
        for sd in seeds:
            rec = traj_grad(J, m, cap, sd)
            v.append(rec[:, 1].min())
        fin[m] = (float(np.mean(v)), float(np.min(v)))
    v = []
    for sd in seeds:
        rec = traj_mom(J, gamma, STEPS_M, sd)
        v.append(rec[:, 1].min())
    fin["momentum"] = (float(np.mean(v)), float(np.min(v)))
    return out, fin


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # ---- 1) dev 扫 gamma (预算与 eval 一致: 60 tu = 1200 步) ----
    Jdev = sk_instance(100, 101)
    print("=== dev 实例 seed=101, N=100, dt=0.05, 1200步 (60tu), 8 种子 ===")
    gscan = {}
    for gamma in (0.0, 0.1, 0.3, 0.5, 1.0):
        vals = []
        for sd in range(8):
            _, rec, _ = momentum_run(Jdev, STEPS_M, dt=DT_M, gamma=gamma, seed=sd)
            vals.append(rec[:, 1].min())
        gscan[gamma] = (float(np.mean(vals)), float(np.min(vals)))
        print(f"  gamma={gamma}: mean {np.mean(vals):+.4f} best {np.min(vals):+.4f}")
    gamma_star = min(gscan, key=lambda g: gscan[g][0])   # 能量越低越好
    print(f"-> 选定 gamma* = {gamma_star} (dev 上 mean 最优)\n")

    # ---- 2) eval ----
    seeds = list(range(0, 8))
    results = {}
    for N in (100, 200):
        thr_list = [-0.68, -0.70, -0.72] if N == 100 else [-0.70, -0.72, -0.74]
        results[N] = {}
        print(f"=== N={N}, 等物理时间 {TU:.0f}tu (梯度 {STEPS_G} 步, 动量 {STEPS_M} 步), "
              f"实例 102-105, 8 种子 ===")
        for inst in (102, 103, 104, 105):
            J = sk_instance(N, inst)
            ttsd, fin = run_instance(J, gamma_star, thr_list, seeds)
            results[N][inst] = {"tts": ttsd, "fin": fin}
            for thr in thr_list:
                def med(m):
                    ok = [t for t in ttsd[m][thr] if t is not None]
                    return float(np.median(ok)) if ok else None, len(ok) / len(seeds)
                gt, gr = med("temp"); mm, mr = med("momentum"); gg, gr2 = med("gain")
                def fmt(v, rate):
                    return f"{v:>5.0f}({rate:.0%})" if v is not None else f"  -- ({rate:.0%})"
                print(f"  thr {thr:+.2f}: 增益 {fmt(gg, gr2)} | 温度 {fmt(gt, gr)} | "
                      f"动量 {fmt(mm, mr)}")
            print(f"   终态(60tu) mean/best: 增益 {fin['gain'][0]:+.4f}/{fin['gain'][1]:+.4f} "
                  f"| 温度 {fin['temp'][0]:+.4f}/{fin['temp'][1]:+.4f} "
                  f"| 动量 {fin['momentum'][0]:+.4f}/{fin['momentum'][1]:+.4f}")
        print()
    json.dump({"gamma_star": gamma_star, "gscan": gscan, "results": results},
              open("honest_rerun.json", "w"), indent=2)
    print("saved honest_rerun.json")
