"""Phase 2 汇总 v2: digCIM 论文配方(5000步) vs smomentum 最优配置(2000步)
终态口径, 同种子; Ps=0 时以 best/mean 为主要对比口径。"""
import json

r = json.load(open("phase2_gset_results.json"))

def best_sm(d):
    best_key, best_v = None, None
    for key, v in d["smomentum"].items():
        f = v["final"]
        if best_v is None or (f["ps"], f["mean"], f["best"]) > (best_v["ps"], best_v["mean"], best_v["best"]):
            best_key, best_v = key, f
    return best_key, best_v

print(f"{'实例':>5} {'known':>6} | {'digCIM best/mean':>16} {'Ps':>6} | "
      f"{'sm最优配置':>12} {'sm best/mean':>14} {'Ps':>6} | {'Δbest':>6} {'Δmean':>7} | TTS比")
print("-" * 100)
wins_b, wins_m = 0, 0
for g in sorted(r, key=lambda x: int(x[1:])):
    d = r[g]
    k = d["known"]
    df = d["digCIM_final"]
    bk, bv = best_sm(d)
    db = bv["best"] - df["best"]
    dm = bv["mean"] - df["mean"]
    wins_b += db > 0
    wins_m += dm > 0
    ratio = None
    if df["tts_steps"] and bv["tts_steps"]:
        ratio = df["tts_steps"] / bv["tts_steps"]
    rt = f"{ratio:.1f}x" if ratio else "—"
    print(f"{g:>5} {k:>6} | {df['best']:>7}/{df['mean']:>7.1f} {df['ps']:>6.0%} | "
          f"{bk:>12} {bv['best']:>7}/{bv['mean']:>7.1f} {bv['ps']:>6.0%} | "
          f"{db:+6} {dm:+7.1f} | {rt}")
n = len(r)
print("-" * 100)
print(f"smomentum 在 best 上胜出 {wins_b}/{n} 实例, mean 上胜出 {wins_m}/{n} 实例 "
      f"(同种子, 2000步 vs 5000步)")
