import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt


# ======================================================
# Utils
# ======================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def soft_update_params(net, target_net, tau: float):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def smooth_curve(y, alpha: float = 0.02):
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return y
    out = np.zeros_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


# ======================================================
# PixelEncoder
# ======================================================
class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward_conv(self, obs):
        obs = obs / 255.0
        x = torch.relu(self.convs[0](obs))
        for conv in self.convs[1:]:
            x = torch.relu(conv(x))
        return x.view(x.size(0), -1)

    def forward(self, obs):
        h = self.forward_conv(obs)
        h = self.fc(h)
        return self.ln(h)


# ======================================================
# PSD MetricNet (ours)
# ======================================================

def rehu(x, delta=1.0):
    return torch.where(
        x <= 0,
        torch.zeros_like(x),
        torch.where(x < delta, x.pow(2) / (2 * delta), x - delta / 2),
    )


class MetricNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, (in_dim * (in_dim + 1)) // 2),
        )
        self.in_dim = in_dim

    def forward(self, z1, z2):
        B = z1.size(0)
        device = z1.device

        def build_G(a, b):
            z_concat = torch.cat([a, b], dim=1)
            L_vec = self.net(z_concat)

            L = torch.zeros(B, self.in_dim, self.in_dim, device=device)
            tril_idx = torch.tril_indices(self.in_dim, self.in_dim, offset=0, device=device)
            L[:, tril_idx[0], tril_idx[1]] = L_vec

            diag_idx = torch.arange(self.in_dim, device=device)
            L[:, diag_idx, diag_idx] = rehu(L[:, diag_idx, diag_idx], delta=1.0)

            return torch.bmm(L, L.transpose(1, 2))

        G1 = build_G(z1, z2)
        G2 = build_G(z2, z1)
        return G1 + G2


def ours_distance(metric_net: nn.Module, z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-8):
    G = metric_net(z1, z2)  
    tr = torch.diagonal(G, dim1=1, dim2=2).sum(dim=1)          
    G = G / (tr.detach().view(-1, 1, 1) + 1e-6)                

    delta = (z1 - z2).unsqueeze(-1)                            
    d2 = torch.bmm(torch.bmm(delta.transpose(1, 2), G), delta).squeeze(-1).squeeze(-1)  
    return torch.sqrt(torch.clamp(d2, min=eps))                


# ======================================================
# DistanceMLP
# ======================================================
class DistanceMLP(nn.Module):
    def __init__(self, in_dim, hidden=392):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x1, x2):
        y = self.net(torch.cat([x1, x2], dim=-1))
        return y.squeeze(-1)


# ======================================================
# Core: fixed-point / Bellman residual fitting (train only)
# ======================================================
def run_fixedpoint_fit_train_only(
    seed: int,
    mode: str,                  # "mlp" or "ours"
    obs: torch.Tensor,          # (N,C,H,W) float in [0,255]
    next_obs: torch.Tensor,     # (N,C,H,W)
    rewards: torch.Tensor,      # (N,1) or (N,)
    device: str,
    z_dim: int = 50,
    gamma: float = 0.99,
    batch_size: int = 256,
    steps: int = 2000,
    log_every: int = 200,
    tau: float = 0.005,
    learning_rate: float = 1e-3,
    freeze_encoder: bool = True,
):
    set_seed(seed)

    # ----- encoder -----
    enc = PixelEncoder(obs.shape[1:], feature_dim=z_dim).to(device)
    if freeze_encoder:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
    else:
        enc.train()
        for p in enc.parameters():
            p.requires_grad = True

    # ----- distance network + target -----
    if mode == "mlp":
        psi = DistanceMLP(z_dim).to(device)

        def dist(net, a, b):
            return net(a, b)

    elif mode == "ours":
        psi = MetricNet(z_dim).to(device)

        def dist(net, a, b):
            return ours_distance(net, a, b)

    else:
        raise ValueError("mode must be 'mlp' or 'ours'")

    psi_target = copy.deepcopy(psi).to(device)
    psi_target.eval()
    for p in psi_target.parameters():
        p.requires_grad = False

    params = list(psi.parameters()) + ([] if freeze_encoder else list(enc.parameters()))
    opt = Adam(params, lr=learning_rate)

    # ----- training indices -----
    N = obs.size(0)
    train_idx = torch.randperm(N, device=device)  # train-only

    # ----- logs -----
    logged_steps = []
    train_losses = []

    for step in range(steps + 1):
        soft_update_params(psi, psi_target, tau)

        # sample transitions
        bidx = train_idx[torch.randint(0, len(train_idx), (batch_size,), device=device)]
        o = obs[bidx]
        no = next_obs[bidx]
        r = rewards[bidx].view(-1)

        # pair within batch
        perm = torch.randperm(batch_size, device=device)
        o2, no2, r2 = o[perm], no[perm], r[perm]

        # encode
        if freeze_encoder:
            with torch.no_grad():
                z, z2 = enc(o), enc(o2)
                z_next, z_next2 = enc(no), enc(no2)
        else:
            z, z2 = enc(o), enc(o2)
            z_next, z_next2 = enc(no), enc(no2)

        # target (no grad)
        with torch.no_grad():
            r_dist = F.smooth_l1_loss(r, r2, reduction="none")
            next_u = dist(psi_target, z_next, z_next2)
            target = r_dist + gamma * next_u

        pred = dist(psi, z, z2)
        loss = (pred - target).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0:
            logged_steps.append(step)
            train_losses.append(loss.item())
            print(f"[{mode}] seed={seed} step={step}/{steps} loss={loss.item():.6f}")

    return np.array(logged_steps), np.array(train_losses)


# ======================================================
# Main: run + plot (train only)
# ======================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # buffer.pt payload list: [obs, next_obs, actions, rewards, curr_rewards, not_dones]
    buf = torch.load("buffer.pt", weights_only=False)
    obs = torch.from_numpy(buf[0]).to(device).float()
    next_obs = torch.from_numpy(buf[1]).to(device).float()
    rewards = torch.from_numpy(buf[3]).to(device).float()

    # (optional) subsample for quick debug
    N_use = 32
    obs = obs[:N_use]
    next_obs = next_obs[:N_use]
    rewards = rewards[:N_use]

    # (optional) reward normalization
    r = rewards.view(-1)
    r = r / (r.max().clamp_min(1e-8))
    rewards = r

    seeds = [1,2,3,4]
    modes = ["ours", "mlp"]

    # hyperparams
    steps = 100000
    log_every = 50
    tau =0.005 # you can choose different number to check the effect of ema
    batch_size = 8
    lr = 1e-4
    gamma = 0.99
    z_dim = 50
    freeze_settings = [True, False]

    for freeze_encoder in freeze_settings:
        results = {m: [] for m in modes}
        for m in modes:
            for s in seeds:
                xs, ys = run_fixedpoint_fit_train_only(
                    seed=s,
                    mode=m,
                    obs=obs,
                    next_obs=next_obs,
                    rewards=rewards,
                    device=device,
                    z_dim=z_dim,
                    gamma=gamma,
                    batch_size=batch_size,
                    steps=steps,
                    log_every=log_every,
                    tau=tau,
                    learning_rate=lr,
                    freeze_encoder=freeze_encoder,
                )
                results[m].append((xs, ys))
                tag = "freeze" if freeze_encoder else "notfreeze"
                np.save(f"raw_{m}_seed{s}_{tag}_train_steps.npy", xs)
                np.save(f"raw_{m}_seed{s}_{tag}_train_loss.npy", ys)


        # ===== plot =====
        plt.figure(figsize=(7, 4))

        title_enc = "Frozen encoder" if freeze_encoder else "Trainable encoder"
        plt.title(f"{title_enc} + psi fixed-point fitting (train)")

        for m, label in [
            ("mlp", "DistanceMLP (fully param)"),
            ("ours", "Structured SPD (ours)"),
        ]:
            # align by steps (assume same logging schedule across seeds)
            step_grid = results[m][0][0]
            loss_mat = np.stack([y for (x, y) in results[m]], axis=0)

            mean_curve = loss_mat.mean(axis=0)
            std_curve = loss_mat.std(axis=0)

            mean_curve = smooth_curve(mean_curve, alpha=0.02)
            std_curve = smooth_curve(std_curve, alpha=0.02)

            plt.plot(step_grid, mean_curve, label=label)
            plt.fill_between(step_grid, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

        plt.xlabel("Training step (updates)")
        plt.ylabel("Bellman residual (MSE)")
        plt.legend()
        plt.tight_layout()
        if freeze_encoder:
            plt.savefig("fixedpoint_train_freeze.png", dpi=200)
            
            print("Saved: fixedpoint_train_freeze.png")
        else:
            plt.savefig("fixedpoint_train_notfreeze.png", dpi=200)
            print("Saved: fixedpoint_train_notfreeze.png")
        plt.close()

