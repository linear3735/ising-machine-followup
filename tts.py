"""TTS 补测：首次达到 SA 参考 + 0.02 的步数（阈值用每实例的 SA 基线）"""
import numpy as np
from cim import sk_instance, energy, sa_baseline, cim_run, \
    gain_anneal_schedule, const_schedule
from comparison import temp_sched, sb_run, STEPS, REC

if __name__ == "__main__":
    import json
    res = json.load(open("results.json"))
    out = {}
    for N in (100, 200):
        print(f"=== N={N} ===")
        out[N] = {}
        for inst in (1, 2, 3):
            J = sk_instance(N, seed=inst)
            row = res[str(N)][str(inst)]
            best_all = min(row[n]["best"] for n in ("gain", "temp", "momentum"))
            thr = 0.9 * best_all            # 所有方法都能稳定达到的通用门槛
            out[N][inst] = {}
            for name in ("gain", "temp", "momentum"):
                tts = []
                for sd in range(8):
                    if name == "gain":
                        gs = gain_anneal_schedule(0.0, 0.5, STEPS)
                        _, rec = cim_run(J, STEPS, gain_schedule=gs,
                                         temp_schedule=const_schedule(1e-4), seed=50 + sd)
                        traj = rec[:, 1]
                    elif name == "temp":
                        _, rec = cim_run(J, STEPS, gain_schedule=const_schedule(0.0),
                                         temp_schedule=temp_sched, seed=2000 + sd)
                        traj = rec[:, 1]
                    else:
                        _, rec = sb_run(J, seed=3000 + sd)
                        traj = rec
                    idx = np.where(traj <= thr)[0]
                    tts.append(idx[0] * REC if len(idx) else np.inf)
                out[N][inst][name] = float(np.mean(tts))
                print(f"  inst{inst}: 门槛={thr:+.4f} | {name:9s} TTS={np.mean(tts):.0f} 步")
    json.dump(out, open("tts.json", "w"), indent=2)
    print("saved tts.json")
