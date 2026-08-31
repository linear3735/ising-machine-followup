"""聚合 results.json + tts.json -> 终版数据表 + 图"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("results.json"))
tts = json.load(open("tts.json"))

METHODS = [("gain", "gain annealing", "#999"),
           ("temp", "temperature annealing\n(Wong et al.)", "#1f77b4"),
           ("closed", "closed-loop residual\n(ours)", "#d62728"),
           ("momentum", "momentum dynamics\n(ours, Wong's future work)", "#ff7f0e")]

agg = {}
for N in ("100", "200"):
    agg[N] = {}
    for key, _, _ in METHODS:
        means = [res[N][i][key]["mean"] for i in res[N]]
        bests = [res[N][i][key]["best"] for i in res[N]]
        stds = [res[N][i][key]["std"] for i in res[N]]
        agg[N][key] = dict(mean=np.mean(means), best=min(bests),
                           std=np.mean(stds))

print("=" * 70)
for N in ("100", "200"):
    print(f"N={N}")
    for key, name, _ in METHODS:
        a = agg[N][key]
        t = tts[N]
        tvals = [t[i][key] for i in t] if key != "closed" else []
        tstr = (f"TTS={np.mean(tvals):.0f}" if tvals and all(v < 1e9 for v in tvals)
                else ("TTS=inf" if tvals else "TTS=—"))
        print(f"  {name:42s}: mean={a['mean']:+.4f}  best={a['best']:+.4f}  {tstr}")

# ---- 图1: 能量对比 (按 N 分组) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, N in zip(axes, ("100", "200")):
    names = [n for _, n, _ in METHODS]
    means = [agg[N][k]["mean"] for k, _, _ in METHODS]
    stds = [agg[N][k]["std"] for k, _, _ in METHODS]
    colors = [c for _, _, c in METHODS]
    ax.bar(range(4), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("decoded energy eps (lower is better)")
    ax.set_title(f"SK N={N}, 3 instances x 8 seeds")
    ax.axhline(-0.763, color="k", ls="--", lw=1)
    ax.text(0.02, -0.764, "Parisi limit -0.763", fontsize=7, va="top")
fig.tight_layout()
fig.savefig("fig_compare.png", dpi=140)
plt.close(fig)

# ---- 图2: TTS 对比 (log scale) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, N in zip(axes, ("100", "200")):
    bars = []
    for idx, (key, name, c) in enumerate(METHODS):
        if key == "closed":
            continue
        vals = [tts[N][i][key] for i in tts[N]]
        if any(v > 1e9 for v in vals):
            pass
        else:
            bars.append((idx, np.mean(vals), c))
    xs = [b[0] for b in bars]
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars])
    ax.set_xticks(xs)
    ax.set_xticklabels([list(dict([(k, n) for k, n, _ in METHODS])).index(
        list(dict([(k, n) for k, n, _ in METHODS]))[x]) for x in [b[0] for b in bars]],
        rotation=0)
    ax.set_xticklabels(["gain\n(not reached)" if b[0] == 0 else
                        "temp\n(Wong)" if b[0] == 1 else "momentum\n(ours)"
                        for b in bars])
    ax.set_ylabel("TTS (steps to 90% frontier)")
    ax.set_yscale("log")
    ax.set_title(f"Convergence speed, N={N}")
fig.tight_layout()
fig.savefig("fig_tts.png", dpi=140)
plt.close(fig)
print("saved fig_compare.png, fig_tts.png")
