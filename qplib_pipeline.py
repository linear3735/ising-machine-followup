"""
QPLIB QUBO 23 题流水线 v2 —— 使用官方解析器 pyqplib（口径零争议）
在租用的多核 CPU 服务器上运行（RunPod CPU pod 即可）。

用法:
  python qplib_pipeline.py --problems 3650,3693,3877 --budget 3600

每题:
  1. 下载 .qplib, 用 pyqplib.read_problem 解析（官方口径: 0.5xᵀQx + bᵀx + q⁰）
  2. 贪心多起点 -> digCIM(论文配方, 热启动/噪声两种初值, 多起点, best-tracking)
  3. 与 GT 对比（GT 表从 Mittelmann 的 QPLIB-QUBO 页抄入 gt.json）
  4. 汇总"我们是否解到GT" vs digCIM 论文 19/23

依赖: pip install numpy scipy pyqplib
"""
import argparse
import json
import os
import ssl
import time
import urllib.request

import numpy as np
import pyqplib

# digCIM 论文 (arXiv:2507.08533 SM Table 8) 逐题成绩: 't'=解到GT, 't-2'/'t-4'=差2/差4
DIGCIM = {
    3506: "t", 3565: "t", 3642: "t", 3650: "t-2", 3693: "t-2",
    3705: "t", 3706: "t", 3738: "t", 3745: "t", 3822: "t",
    3832: "t-4", 3838: "t-4", 3850: "t-4", 3852: "t", 3877: "t-2",
    5721: "t", 5725: "t", 5755: "t", 5875: "t",
    5881: "t", 5882: "t", 5909: "t", 5922: "t",
}
# 优先攻击名单: digCIM 未到 GT 的 6 题
PRIORITY = [3650, 3693, 3877, 3832, 3838, 3850]


def download(url, path):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx, timeout=120) as resp, \
            open(path, "wb") as f:
        f.write(resp.read())


def greedy(p, seed):
    n = p.num_vars
    r = np.random.default_rng(seed)
    x = np.ones(n)
    for i in r.permutation(n):
        g = p.obj.grad(x)
        if (1.0 - 2.0 * x[i]) * g[i] > 0:
            x[i] = 1.0 - x[i]
    return x


def digcim(p, steps, dt, a, T0, seed, warm=False):
    n = p.num_vars
    r = np.random.default_rng(seed)
    z = 2.0 * greedy(p, seed) - 1.0 if warm else r.normal(0, 0.01, n)
    best = -1e18
    for t in range(steps):
        T = T0 * (1.0 - t / max(1, steps - 1))
        x = (np.sign(z) + 1.0) / 2.0
        z = z + dt * (a * z + 0.5 * p.obj.grad(x))
        z = z + np.sqrt(2.0 * T * dt) * r.standard_normal(n)
        z = np.clip(z, -1.0, 1.0)
        if t % 100 == 0 or t == steps - 1:
            best = max(best, p.obj.eval((np.sign(z) + 1.0) / 2.0))
    return best


def tabu(p, seed, iters, tenure=20, pert_after=2000):
    """Tabu 搜索（经典局部搜索, 最大化 0.5xQx+bx）+ 停滞扰动"""
    n = p.num_vars
    r = np.random.default_rng(seed)
    x = np.ones(n)
    best = p.obj.eval(x)
    bestx = x.copy()
    tabu = np.zeros(n, int)
    since = 0
    for it in range(iters):
        g = np.array(p.obj.grad(x))
        delta = (1.0 - 2.0 * x) * g
        # 选允许的最佳翻转: 优先改进, 否则非禁忌中挑损失最小的
        cand = [i for i in range(n) if delta[i] > 1e-9 and tabu[i] <= it]
        if not cand:
            cand = [i for i in range(n) if tabu[i] <= it]
            if not cand:
                break
        i = max(cand, key=lambda j: delta[j])
        x[i] = 1.0 - x[i]
        tabu[i] = it + tenure
        v = p.obj.eval(x)
        if v > best:
            best, bestx, since = v, x.copy(), 0
        else:
            since += 1
        if since > pert_after:          # 长期停滞 -> 扰动(跳到贪心解附近)
            x = greedy(p, seed + it).copy()
            tabu[:] = 0
            since = 0
    return best


def solve(pid, budget):
    f = f"QPLIB_{pid}.qplib"
    download(f"https://qplib.zib.de/qplib/{f}", f)
    p = pyqplib.read_problem(f)
    t0 = time.time()
    gbest = max(p.obj.eval(greedy(p, s)) for s in range(50))
    dbest, tb, runs = -1e18, -1e18, 0
    while time.time() - t0 < budget * 0.6:
        steps = 5000 if p.num_vars <= 2000 else 3000
        dbest = max(dbest, digcim(p, steps, 0.03, -10.0, 3.0,
                                  100000 + runs, warm=(runs % 2 == 0)))
        runs += 1
    while time.time() - t0 < budget * 0.98:
        tb = max(tb, tabu(p, 700000 + runs, iters=4000))
        runs += 1
    return dict(n=p.num_vars, 贪心=round(gbest, 2), digCIM_best=round(dbest, 2),
                tabu_best=round(tb, 2), runs=runs,
                耗时=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=str, default=",".join(map(str, PRIORITY)))
    ap.add_argument("--budget", type=float, default=3600)
    args = ap.parse_args()
    pids = [int(x) for x in args.problems.split(",")]
    gt = json.load(open("gt.json")) if os.path.exists("gt.json") else {}
    out = {}
    for pid in pids:
        try:
            r = solve(pid, args.budget)
            r["GT"] = gt.get(str(pid))
            r["digCIM论文"] = DIGCIM.get(pid, "?")
            if r["GT"] is not None:
                r["我们是否解到GT"] = bool(
                    max(r["digCIM_best"], r["tabu_best"]) >= r["GT"] - 1e-6)
            out[pid] = r
            print(pid, r, flush=True)
        except Exception as e:
            out[pid] = {"error": str(e)}
            print(pid, "错误:", e, flush=True)
    json.dump(out, open("qplib_results.json", "w"), indent=2)
    solved = sum(1 for v in out.values() if v.get("我们是否解到GT"))
    print(f"\n汇总: {solved}/{len(pids)} 解到GT（digCIM 论文: 19/23）")
