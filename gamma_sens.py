"""γ 敏感性: 同 honest_rerun 协议, γ ∈ {0.1, 0.3, 1.0}, N=100/200, 实例 102-105
等物理时间 60tu, 物理时间口径 TTS (排除 dt 差异)。"""
import numpy as np
from cim import sk_instance, cim_run, const_schedule
from momentum import momentum_run

DT_G, DT_M, TU = 0.02, 0.05, 60.0
SG, SM = int(TU / DT_G), int(TU / DT_M)


def temp_sched(t, cap):
    return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / cap)


def tts(rec, thr):
    idx = np.where(rec[:, 1] <= thr)[0]
    return int(rec[idx[0], 0]) if len(idx) else None


for N in (100, 200):
    thrs = [-0.68, -0.70, -0.72] if N == 100 else [-0.70, -0.72, -0.74]
    print(f"=== N={N} ===")
    for inst in (102, 103, 104, 105):
        J = sk_instance(N, inst)
        line = [f"inst{inst}"]
        for thr in thrs:
            tv = []
            for sd in range(8):
                _, rec = cim_run(J, SG, gain_schedule=const_schedule(0.0),
                                 temp_schedule=lambda t: temp_sched(t, SG), seed=sd)
                tv.append(tts(rec, thr))
            ok = [t * DT_G for t in tv if t is not None]
            tmed = float(np.median(ok)) if ok else None
            tr = len(ok) / 8
            parts = []
            for g in (0.1, 0.3, 1.0):
                mv = []
                for sd in range(8):
                    _, rec, _ = momentum_run(J, SM, dt=DT_M, gamma=g, seed=sd)
                    mv.append(tts(rec, thr))
                okm = [t * DT_M for t in mv if t is not None]
                mmed = float(np.median(okm)) if okm else None
                mr = len(okm) / 8
                ratio = (tmed / mmed) if (tmed and mmed) else None
                if mmed is None:
                    parts.append(f"g{g}: -- ({mr:.0%})")
                elif ratio:
                    parts.append(f"g{g}:{mmed:.0f}tu({mr:.0%})x{ratio:.1f}")
                else:
                    parts.append(f"g{g}:{mmed:.0f}tu({mr:.0%})")
            Tstr = f"T:{tmed:.0f}tu({tr:.0%})" if tmed is not None else f"T: -- ({tr:.0%})"
            line.append(f"thr{thr:+.2f} {Tstr} | " + " ".join(parts))
        print("  " + " | ".join(line), flush=True)
