"""killtest2.py —— z-潜空间草案 vs σ-随机翻转草案 (结构假设的击杀测试)

核心测量 (held-out 实例):
  1. 结构曲线: 草案能量 vs 与参考解的 Hamming 距离 —— flow 若学了低能流形,
     z-草案应随 Hamming 增大保持低能, 而翻转草案能量线性劣化;
  2. 注入价值: 短精修 (300 步低T polish) 后的能量 vs Hamming ——
     同扰动幅度下 z-草案是否精修到更低能量 / 更高逃逸率。
"""
import numpy as np
import torch
from cim import sk_instance
from realnvp_ising import RealNVP

DIM = 100


def eps_sk(J, sgn):
    return (sgn * (sgn @ J)).sum(axis=1) / (2.0 * J.shape[0])


def polish(J, x0s, steps=300, dt=0.2, gamma=3.0, a=-10.0, T0=0.05, seed0=5):
    """短精修: 从草案 x0 出发的低温 smomentum (T: T0->1e-4 线性)。返回 (终态eps, 全程best eps)"""
    S = x0s.shape[0]
    N = J.shape[0]
    rngs = [np.random.default_rng(seed0 + s) for s in range(S)]
    x = np.clip(x0s, -1.0, 1.0).copy()
    y = np.array([r.normal(0, 0.01, N) for r in rngs])
    sgn = np.sign(x)
    h = sgn @ J
    best = eps_sk(J, sgn)
    dt_g = 1.0 / (1.0 + dt * gamma)
    for t in range(steps):
        T = max(T0 * (1.0 - t / max(1, steps - 1)), 1e-4)
        ns = np.sqrt(2.0 * T * dt)
        y *= dt_g
        y += (dt * a * x - dt * h) * dt_g
        for s in range(S):
            y[s] += ns * rngs[s].standard_normal(N)
        x += dt * y
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
        sgn = np.sign(x)
        h = sgn @ J
        if t % 50 == 0 or t == steps - 1:
            e = eps_sk(J, sgn)
            best = np.minimum(best, e)
    return eps_sk(J, sgn), best


def hamming(xa, xb):
    return (np.sign(xa) != np.sign(xb)).mean(axis=1)


def main():
    np.set_printoptions(suppress=True)
    torch.manual_seed(0)
    model = RealNVP(DIM, n_layers=6, hidden=128)
    model.load_state_dict(torch.load("flow_sk100.pt", weights_only=True))
    model.eval()

    for inst in (200, 201, 202):
        J = sk_instance(DIM, inst)
        N = DIM
        # 参考好解: 32 种子 smomentum 全退火取 best 终态 x
        from smomentum_fast import smomentum_batch
        sgn32, _ = smomentum_batch(J, 2000, 0.2, 3.0, -10.0, 3.0, list(range(0, 32)))
        e32 = eps_sk(J, sgn32)
        i0 = int(np.argmin(e32))
        x_ref = (sgn32[i0].astype(np.float32)
                 + 0.02 * np.random.default_rng(inst).standard_normal(N).astype(np.float32))
        e_ref = e32[i0]
        z_ref = model.forward(torch.tensor(x_ref[None]))[0].detach().numpy()[0]

        print(f"\n=== SK N=100 实例 {inst}: 参考好解 E={e_ref:+.4f} "
              f"(32种子 best) ===")
        print(f"{'草案族':>10} {'扰动':>6} {'mean Ham':>8} {'草案E(mean/best)':>18} "
              f"{'精修后E(mean/best)':>18} {'逃逸率':>7}")

        rows = []
        K = 64
        # A) z-潜空间草案
        for eps_z in (0.05, 0.2, 0.6, 1.2):
            r = np.random.default_rng(inst * 100 + int(eps_z * 100))
            z = z_ref[None] + eps_z * r.standard_normal((K, N))
            xd = model.inverse(torch.tensor(z, dtype=torch.float32)).detach().numpy()
            xd = np.clip(xd, -1, 1)
            ed = eps_sk(J, np.sign(xd))
            ef, eb = polish(J, xd)
            esc = (eb < e_ref - 1e-3).mean()
            rows.append(("z草案", eps_z, hamming(xd, x_ref[None]).mean(), ed, ef, eb, esc))
        # B) σ-随机翻转草案
        for k in (1, 5, 15, 40):
            r = np.random.default_rng(inst * 7 + k)
            xd = np.tile(x_ref[None], (K, 1))
            for s in range(K):
                idx = r.choice(N, k, replace=False)
                xd[s, idx] = -xd[s, idx]
            xd = np.clip(xd + 0.02 * r.standard_normal((K, N)), -1, 1)
            ed = eps_sk(J, np.sign(xd))
            ef, eb = polish(J, xd)
            esc = (eb < e_ref - 1e-3).mean()
            rows.append(("翻转", k, k / N, ed, ef, eb, esc))
        # C) 噪声基线
        r = np.random.default_rng(inst)
        xd = np.clip(r.normal(0, 0.3, (K, N)), -1, 1)
        ed = eps_sk(J, np.sign(xd))
        ef, eb = polish(J, xd)
        esc = (eb < e_ref - 1e-3).mean()
        rows.append(("噪声", "-", 0.5, ed, ef, eb, esc))

        for fam, pert, ham, ed, ef, eb, esc in rows:
            print(f"{fam:>10} {pert:>6} {ham:>8.3f} "
                  f"{ed.mean():+.4f}/{ed.min():+.4f} "
                  f"{ef.mean():+.4f}/{ef.min():+.4f} {esc:>7.0%}")


if __name__ == "__main__":
    main()
