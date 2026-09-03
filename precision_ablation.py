"""precision_ablation.py —— 精度消融 + 舍入噪声理论迁移测试 (严谨验证)

A. 实证消融: 论文主角梯度温度退火 (a=0, T:0.5->1e-4) 在 SK 上
   FP64/FP32/FP16(native)/FP16(逐元素舍入仿真) 动力学, 解码恒用 FP64;
   附 digCIM G1 的 FP64/FP32/FP16 消融 (Ps 到 known)。
B. 理论迁移: FP64 动力学注入受控相对舍入噪声 sigma (逐元素乘性), 
   测解码能量偏差 vs sigma —— 若 bias ~ sigma^2 (NQS 界的 SK 版),
   则 FP32 (sigma~3e-8) 与 FP16 (sigma~3e-4) 的影响可定量预测。
"""
import numpy as np
import time
from cim import sk_instance, cim_run, const_schedule
from gset_bench import load_gset, digcim_run

DT = 0.02
STEPS = 2000
SEEDS = 8


def temp_sched(t, cap=STEPS):
    return 1e-4 + (0.5 - 1e-4) * max(0.0, 1.0 - t / cap)


def energy64(J, s):
    return (s.astype(np.float64) @ J.astype(np.float64)
            @ s.astype(np.float64)) / (2.0 * J.shape[0])


def run_sk_prec(J, dtype, seed, noise_sigma=None, round16=False):
    """梯度温度退火 at 指定精度。noise_sigma: 注入舍入噪声 (B 测试);
    round16: 每步经 float16 舍入往返 (真 FP16 数据通路仿真)。"""
    N = J.shape[0]
    Jp = J.astype(dtype)
    r = np.random.default_rng(seed)
    x = r.normal(0, 0.01, N).astype(dtype)
    best = 1e9
    for t in range(STEPS):
        T = temp_sched(t)
        x = x + DT * (-x ** 3 - (Jp @ x))
        x = x + np.sqrt(2.0 * T * DT) * r.standard_normal(N).astype(dtype)
        if noise_sigma is not None:
            x = x * (1.0 + noise_sigma * r.standard_normal(N).astype(dtype))
        if round16:
            x = x.astype(np.float16).astype(dtype)
        if t % 50 == 0 or t == STEPS - 1:
            e = energy64(J, np.sign(x))
            best = min(best, e)
    return best


def ablation_sk():
    print("=== A1. SK 梯度温度退火精度消融 (mean/best over 实例x种子, 解码 FP64) ===")
    for N in (100, 200):
        for dtype, tag in ((np.float64, "FP64"), (np.float32, "FP32"),
                           (np.float16, "FP16-native")):
            vals = []
            for inst in (101, 102, 103, 104):
                J = sk_instance(N, inst)
                for sd in range(SEEDS):
                    vals.append(run_sk_prec(J, dtype, sd))
            print(f"  N={N:>3} {tag:>11}: mean={np.mean(vals):+.5f} "
                  f"best={np.min(vals):+.5f} (n={len(vals)})")
        # FP16 仿真 (float16 逐元素往返), 只 N=100
        if N == 100:
            vals = []
            for inst in (101, 102, 103, 104):
                J = sk_instance(N, inst)
                for sd in range(SEEDS):
                    vals.append(run_sk_prec(J, np.float32, sd, round16=True))
            print(f"  N={N:>3} {'FP16-emul':>11}: mean={np.mean(vals):+.5f} "
                  f"best={np.min(vals):+.5f} (n={len(vals)})")


def ablation_gset():
    print("\n=== A2. digCIM G1 精度消融 (Ps >= 11624, 32 种子) ===")
    A = load_gset("G1.dat")
    for dtype, tag in ((np.float64, "FP64"), (np.float32, "FP32"),
                       (np.float16, "FP16")):
        hits, vals = 0, []
        for sd in range(32):
            s, b = digcim_run(A.astype(dtype), 5000, 0.03, -10.0, 3.0, sd)
            c = (A.sum() / 4.0) - (s.astype(np.float64) @ A.astype(np.float64)
                                   @ s.astype(np.float64)) / 4.0
            hits += c >= 11624
            vals.append(c)
        print(f"  {tag:>9}: Ps={hits/32:.0%} mean={np.mean(vals):.0f} "
              f"best={np.max(vals):.0f}")


def noise_transfer():
    print("\n=== B. 舍入噪声迁移测试: FP64 动力学注入 sigma, 偏差 vs sigma ===")
    J = sk_instance(100, 101)
    ref = np.mean([run_sk_prec(J, np.float64, sd) for sd in range(SEEDS)])
    print(f"  参考 (sigma=0): mean={ref:+.5f}")
    print(f"  {'sigma':>8} {'bias':>10} {'log-log斜率(σ²检验)':>14}")
    prev = None
    for sigma in (1e-7, 1e-5, 1e-4, 1e-3, 1e-2):
        vals = [run_sk_prec(J, np.float64, sd, noise_sigma=sigma)
                for sd in range(SEEDS)]
        bias = float(np.mean(vals) - ref)
        slope = np.log(bias / prev) / np.log(sigma / (sigma / 10)) if prev else None
        prev = bias
        s = f"{slope:+.2f}" if slope else "  —"
        print(f"  {sigma:>8.0e} {bias:>+10.5f} {s:>14}")
    # 预测: FP32 sigma~3e-8 -> bias ~ b(1e-5)·(3e-3)^2; FP16 sigma~3e-4
    print("  若 bias~σ²: FP32(σ≈3e-8) 偏差 ≈ b(1e-5)×1e-6 ≈ 不可测; "
          "FP16(σ≈3e-4) 偏差 ≈ b(1e-5)×900 ≈ 0.01×b(1e-3)")


if __name__ == "__main__":
    t0 = time.time()
    ablation_sk()
    ablation_gset()
    noise_transfer()
    print(f"\n总耗时 {time.time()-t0:.0f}s")
