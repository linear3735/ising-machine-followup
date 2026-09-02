"""realnvp_ising.py —— SK N=100 好解流形的小型 RealNVP (方向 B / kill-test 2 的生成模型)

- 数据: flow_data.npz (smomentum 终态 x ∈[-1,1], 能量过滤, 抖动);
- 架构: 6 层仿射耦合 (mask 交替), 隐藏 128; 训练 = 最大似然 (NLL);
- 用途: (a) 采样新草案; (b) 解析逆映射 z* = f^{-1}(x*) 做潜空间扰动。
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden=128):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2 * dim))

    def forward(self, x):
        mx = x * self.mask
        out = self.net(mx)
        s, t = out.chunk(2, dim=-1)
        s = torch.tanh(s) * 1.5               # 有界 log-scale, 稳定训练
        inv = 1.0 - self.mask
        y = mx + inv * (x * torch.exp(s) + t)
        logdet = (inv * s).sum(dim=-1)
        return y, logdet

    def inverse(self, y):
        my = y * self.mask
        out = self.net(my)
        s, t = out.chunk(2, dim=-1)
        s = torch.tanh(s) * 1.5
        inv = 1.0 - self.mask
        return my + inv * ((y - t) * torch.exp(-s))


class RealNVP(nn.Module):
    def __init__(self, dim, n_layers=6, hidden=128):
        super().__init__()
        self.layers = nn.ModuleList()
        for L in range(n_layers):
            m = [(i + L) % 2 == 0 for i in range(dim)]
            self.layers.append(AffineCoupling(dim, [1.0 if b else 0.0 for b in m],
                                              hidden=hidden))

    def forward(self, x):
        logdet = torch.zeros(x.shape[0])
        for lay in self.layers:
            x, ld = lay(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, z):
        x = z
        for lay in reversed(self.layers):
            x = lay.inverse(x)
        return x

    def nll(self, x):
        z, logdet = self.forward(x)
        return 0.5 * (z ** 2).sum(dim=-1) - logdet


def train_flow(x, epochs=400, lr=1e-3, bs=64, hidden=128, n_layers=6, seed=0):
    torch.manual_seed(seed)
    dim = x.shape[1]
    model = RealNVP(dim, n_layers=n_layers, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(x, dtype=torch.float32)
    n = len(X)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = X[perm[i:i + bs]]
            loss = model.nll(b).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        hist.append(tot / n)
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"  ep {ep:4d}: nll {hist[-1]:.2f}", flush=True)
    return model, np.array(hist)


if __name__ == "__main__":
    d = np.load("flow_data.npz")
    x = d["x"]; e = d["e"]
    print(f"数据: {x.shape}, 能量范围 [{e.min():+.4f}, {e.max():+.4f}]")
    model, hist = train_flow(x, epochs=500)
    torch.save(model.state_dict(), "flow_sk100.pt")
    print("saved flow_sk100.pt")
    # 自检: 训练集内逆映射重建误差 (解析逆应 ~0)
    X = torch.tensor(x[:50], dtype=torch.float32)
    z, _ = model.forward(X)
    xr = model.inverse(z)
    print(f"重建误差: {(xr - X).abs().max():.2e}")
    # 自检: 采样多样性
    xs = model.inverse(torch.randn(200, x.shape[1])).detach().numpy()
    print(f"采样: 200 个, 不同符号构型 {(np.sign(xs) != np.sign(xs[0])).any(axis=1).sum()} 个")
