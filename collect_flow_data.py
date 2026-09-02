"""collect_flow_data.py —— 收集 smomentum 在 SK N=100 上的终态 x 数据 (方向B数据源)

动力学: smomentum_digcim 作用于 SK 耦合 J (digCIM 驱动 + 阻尼动量 + 温度 + 墙),
能量: eps = s^T J s / (2N) (最小化, Parisi 极限 ~ -0.763)。
输出: flow_data.npz {x: (M,N) 终态连续状态(加微抖动), e: (M,) 能量}
"""
import numpy as np
import time
from cim import sk_instance
from smomentum_fast import smomentum_batch


def eps_sk(J, sgn):
    return (sgn * (sgn @ J)).sum(axis=1) / (2.0 * J.shape[0])


if __name__ == "__main__":
    N = 100
    SEEDS = list(range(0, 64))
    train_ids = list(range(1, 17))     # 训练实例 1-16 (与 closed_loop 同协议)
    X, E = [], []
    t0 = time.time()
    for sid in train_ids:
        J = sk_instance(N, sid)
        sgn, best = smomentum_batch(J, 2000, 0.2, 3.0, -10.0, 3.0, SEEDS)
        e = eps_sk(J, sgn)
        X.append(sgn)
        E.append(e)
        print(f"实例 {sid}: mean {e.mean():+.4f} best {e.min():+.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    X = np.concatenate(X)              # (1024, 100) 连续终态在墙 ±1
    E = np.concatenate(E)
    # 能量过滤 + 抖动 (去重 + 平滑 ±1 尖峰)
    thr = np.percentile(E, 40)         # 保留最好 40%
    keep = E <= thr
    xf = X[keep] + 0.02 * np.random.default_rng(0).standard_normal((keep.sum(), N))
    ef = E[keep]
    print(f"\n过滤: {len(E)} -> {len(ef)} 样本, 阈值 {thr:+.4f}")
    np.savez("flow_data.npz", x=xf.astype(np.float32), e=ef)
    print("saved flow_data.npz")
