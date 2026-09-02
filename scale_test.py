"""scale_test v2 —— 动量 vs 梯度温度退火 TTS 随 N 的标度 (无 SA 依赖)

阈值: thr = -0.700 固定 (Parisi -0.763 深度的 91.7%, 与 N=100 原协议同水平,
跨 N 一致)。TTS = 首次达到 thr 的步数, 上限 8000。质量: best/mean (6000 步)。
"""
import numpy as np
import time
import json
from cim import sk_instance, cim_run, const_schedule, temp_anneal_schedule
from momentum import momentum_run

THR = -0.700
CAP = 8000


def tts_to_thr(rec, thr):
    for t, e in rec:
        if e <= thr:
            return t
    return np.inf


def one(N, inst, seeds):
    J = sk_instance(N, inst)
    tg, tm, bg, bm = [], [], [], []
    for sd in seeds:
        s_g, rec_g = cim_run(J, CAP, gain_schedule=const_schedule(0.0),
                             temp_schedule=temp_anneal_schedule(0.5, 1e-4, CAP),
                             seed=sd)
        tg.append(tts_to_thr(rec_g, THR))
        bg.append(rec_g[:, 1].min())
        s_m, rec_m, _ = momentum_run(J, CAP, dt=0.05, gamma=0.3, seed=sd)
        tm.append(tts_to_thr(rec_m, THR))
        bm.append(rec_m[:, 1].min())
    med = lambda v: float(np.median([t for t in v if np.isfinite(t)])) \
        if any(np.isfinite(v)) else np.inf
    rate = lambda v: sum(np.isfinite(v)) / len(v)
    return dict(N=N, inst=inst, tts_grad=med(tg), tts_mom=med(tm),
                rate_g=rate(tg), rate_m=rate(tm),
                best_g=float(min(bg)), best_m=float(min(bm)),
                mean_g=float(np.mean(bg)), mean_m=float(np.mean(bm)))


if __name__ == "__main__":
    rows = []
    for N, insts in ((100, (101, 102)), (200, (101, 102)),
                     (500, (101, 102)), (1000, (101, 102))):
        for inst in insts:
            t0 = time.time()
            r = one(N, inst, list(range(0, 8)))
            r["sec"] = round(time.time() - t0)
            rows.append(r)
            rt = (r["tts_grad"] / r["tts_mom"]) if np.isfinite(r["tts_mom"]) else "∞"
            print(f"N={N:>4} inst={inst}: TTS 梯度 {r['tts_grad']:>7.0f} vs "
                  f"动量 {r['tts_mom']:>6.0f} = {rt if isinstance(rt,str) else f'{rt:.1f}x'} "
                  f"| best {r['best_g']:+.4f}/{r['best_m']:+.4f} "
                  f"mean {r['mean_g']:+.4f}/{r['mean_m']:+.4f} "
                  f"(达标率 {r['rate_g']:.0%}/{r['rate_m']:.0%}) [{r['sec']}s]", flush=True)
    json.dump(rows, open("scale_test.json", "w"), indent=2)
    print("\n汇总 (TTS 到 -0.700):")
    for r in rows:
        rt = (r["tts_grad"] / r["tts_mom"]) if np.isfinite(r["tts_mom"]) else None
        print(f"N={r['N']:>4}: {rt:>5.1f}x" if rt else f"N={r['N']:>4}: 未达")
