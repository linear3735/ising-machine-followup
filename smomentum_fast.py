"""smomentum_fast.py v2 —— 修正: 增量局部场只在稠密大实例才划算;
小实例瓶颈是 Python 逐步开销 -> 算子融合 + 批量稀疏 matvec。"""
import numpy as np
import scipy.sparse as sp


def smomentum_batch(A, steps, dt, gamma, a, Tinit, seeds, csr=None,
                    incremental=None):
    """批量化 smomentum_digcim, 与逐种子串行等价（每种子独立 rng）。
    incremental: None=自动 (nnz>1e6 且翻转少时用 csc 列修正, 否则全量 csr matvec)"""
    N = A.shape[0]
    S = len(seeds)
    if csr is None:
        csr = sp.csr_matrix(A) if sp.issparse(A) else sp.csr_matrix(np.asarray(A))
    if incremental is None:
        incremental = csr.nnz > 1e6
    csc = csr.T.tocsc() if incremental else None
    rngs = [np.random.default_rng(sd) for sd in seeds]

    x = np.array([r.normal(0, 0.01, N) for r in rngs])
    y = np.array([r.normal(0, 0.01, N) for r in rngs])
    sgn = np.sign(x)
    h = np.asarray(csr @ sgn.T)                       # (N, S)
    best = np.full(S, -1e18)
    m = csr.sum() / 2.0
    dt_g = 1.0 / (1.0 + dt * gamma)
    a_dt = dt * a
    for t in range(steps):
        T = Tinit * (1.0 - t / max(1, steps - 1))
        ns = np.sqrt(2.0 * T * dt)
        y *= dt_g
        y += (a_dt * x - dt * h.T) * dt_g
        for s in range(S):                            # 噪声(每种子独立序列)
            y[s] += ns * rngs[s].standard_normal(N)
        x += dt * y
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
        new_sgn = np.sign(x)
        if incremental:
            flips = new_sgn != sgn
            nf = int(flips.sum())
            if nf and nf < 0.05 * S * N:              # 翻转少 -> 稀疏修正
                for s in range(S):
                    idx = np.nonzero(flips[s])[0]
                    if len(idx):
                        h[:, s] += csc[:, idx] @ (new_sgn[s, idx] - sgn[s, idx])
            else:
                h = np.asarray(csr @ new_sgn.T)
        else:
            h = np.asarray(csr @ new_sgn.T)
        sgn = new_sgn
        if t % 100 == 0 or t == steps - 1:
            cuts = m - (sgn * h.T).sum(axis=1) / 4.0
            best = np.maximum(best, cuts)
    return sgn, best


if __name__ == "__main__":
    import time
    from gset_bench import load_gset, cut
    from momentum import smomentum_digcim

    np.set_printoptions(suppress=True)
    seeds = list(range(200, 232))  # 32 种子
    for g, dt_ in (("G11", 0.25), ("G1", 0.2), ("G22", 0.25)):
        A = load_gset(f"{g}.dat")
        t0 = time.time()
        ref = np.array([cut(A, smomentum_digcim(A, 2000, dt_, 3.0, -10.0, 3.0, s))
                        for s in seeds])
        t_ref = time.time() - t0
        t0 = time.time()
        sgn, best = smomentum_batch(A, 2000, dt_, 3.0, -10.0, 3.0, seeds)
        t_batch = time.time() - t0
        cuts = np.array([cut(A, s) for s in sgn])
        ok = (cuts == ref).all()
        print(f"{g:>4} 串行稠密 {t_ref:5.1f}s | 批量+稀疏 {t_batch:5.1f}s "
              f"-> {t_ref/t_batch:4.1f}x  一致={ok}")
