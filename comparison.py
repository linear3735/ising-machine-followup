"""
统一对比实验：SK 基准上四方法 x 能量 x TTS（论文主表）
方法: 增益退火(CIM) / 温度退火(CIM, Wong 解析最优) / 闭环残差策略(本文, N=100 训练)
      / 动量动力学(SB 式, Wong 论文的 future work)
指标: mean ± std, best, TTS95 (首次达到 best_found+0.01 的步数)
"""
import json
import numpy as np
from cim import sk_instance, energy, sa_baseline, cim_run, \
    gain_anneal_schedule, const_schedule
from closed_loop import rollout_cl, unflat

DT = 0.02
STEPS = 3000
REC = 20


def temp_sched(t):
    return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / STEPS)


def sb_run(J, steps=STEPS, dt=0.3, a0=1.0, anneal_frac=0.5, c0=1.0, seed=0):
    """动量动力学 (SB 式), 返回 sign 和轨迹"""
    N = J.shape[0]
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N)
    y = r.normal(0, 0.01, N)
    T_anneal = max(1, int(steps * anneal_frac))
    rec = []
    for t in range(1, steps + 1):
        a = a0 * min(1.0, t / T_anneal)
        y = y + dt * (-(a0 - a) * x - c0 * (J @ x))
        x = x + dt * (a0 * y)
        x = np.tanh(x)
        if t % REC == 0:
            rec.append(energy(J, np.sign(x)))
    return np.sign(x), np.array(rec)


def run_method(J, name, seeds, params_cl=None):
    vals, tts_list = [], []
    for sd in seeds:
        rec = None
        if name == "gain":
            gs = gain_anneal_schedule(0.0, 0.5, STEPS)
            s, rec = cim_run(J, STEPS, gain_schedule=gs,
                             temp_schedule=const_schedule(1e-4), seed=sd)
        elif name == "temp":
            s, rec = cim_run(J, STEPS, gain_schedule=const_schedule(0.0),
                             temp_schedule=temp_sched, seed=sd)
        elif name == "closed":
            sgn, final, _ = rollout_cl([J], STEPS, params_cl, seed=sd)
            s = sgn[0]
        elif name == "momentum":
            s, rec = sb_run(J, seed=sd)
        e = energy(J, s)
        vals.append(e)
        if rec is not None and len(rec) > 0:
            tts_list.append(np.argmin(rec))  # 轨迹最低点出现的位置
    vals = np.array(vals)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "best": float(vals.min()), "tts_min": None}


def tts95(J, name, seeds, target, params_cl=None):
    """首次达到 target+0.01 的步数 (target = 全部方法的最好值), 均值"""
    tts = []
    for sd in seeds:
        rec = None
        if name == "gain":
            gs = gain_anneal_schedule(0.0, 0.5, STEPS)
            _, rec = cim_run(J, STEPS, gain_schedule=gs,
                             temp_schedule=const_schedule(1e-4), seed=sd)
        elif name == "temp":
            _, rec = cim_run(J, STEPS, gain_schedule=const_schedule(0.0),
                             temp_schedule=temp_sched, seed=sd)
        elif name == "momentum":
            _, rec = sb_run(J, seed=sd)
        if rec is not None:
            idx = np.where(rec <= target + 0.01)[0]
            tts.append(idx[0] * REC if len(idx) else np.inf)
    return float(np.mean(tts))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    params_cl = unflat(np.load("theta_cl.npy"))
    results = {}
    for N in (100, 200):
        results[N] = {}
        for inst in (1, 2, 3):
            J = sk_instance(N, seed=inst)
            seeds = [50 + i for i in range(8)]
            row = {}
            for name in ("gain", "temp", "momentum"):
                row[name] = run_method(J, name, seeds)
            row["closed"] = run_method(J, "closed", seeds, params_cl)
            # 目标值: 全方法最好
            target = min(row[n]["best"] for n in row)
            for name in ("gain", "temp", "momentum"):
                row[name]["tts95"] = tts95(J, name, seeds, target)
            print(f"N={N} inst={inst} target={target:+.4f}")
            for name, r in row.items():
                print(f"   {name:9s}: mean={r['mean']:+.4f} ± {r['std']:.4f} "
                      f"best={r['best']:+.4f} tts95={r.get('tts95')}")
            results[N][inst] = row
    json.dump(results, open("results.json", "w"), indent=2, default=float)
    print("saved results.json")
