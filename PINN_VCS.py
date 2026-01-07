# ============================================================
# PINN for VCS
# ============================================================
import os
import math
import time
import re
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Device + seeds
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# PDE / domain parameters
# ============================================================
MU_VISCOSITY   = 1e-3
GAMMA_INDEX    = 1.0
FORCE_F_CONST  = 0.0
STABILITY_EPS  = 1e-7

L_DOMAIN = 1.0
T_DOMAIN = 0.30

# ============================================================
# Global hyperparameters
# ============================================================
LEARNING_RATE_ADAM  = 1e-4
LEARNING_RATE_LBFGS = 1.0

ADAM_EPOCHS_DEFAULT  = 5000   # M1
LBFGS_STEPS_DEFAULT  = 1000   # M2 = max_iter

N_PDE = 8000
N_IC  = 512
N_BC  = 1024

# Loss weights
LAMBDA_PDE1 = 1.0
LAMBDA_PDE2 = 1.0
LAMBDA_PDE3 = 2.0
LAMBDA_IC   = 1.0
LAMBDA_BC   = 1.0
LAMBDA_DX_RHO_IC     = 0.0
LAMBDA_DX_RHOSTAR_IC = 1.0

TEMPERATURE_RHO_DEFAULT = 4.0

# Toggle penalty on dx(rho*)
IC_RS_CONST = False

# ============================================================
# Initial data
# ============================================================
def initial_rho_func(x, l_domain):
    return 0.7 * torch.ones_like(x)
    # case 3:
    # amplitude    = 0.2
    # background   = 0.4
    # pulse_width  = l_domain / 10.0
    # pulse_center = l_domain / 2.0
    # x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    # return background + amplitude * torch.exp(-((x_tensor - pulse_center) ** 2) /
    #                                           (2 * pulse_width ** 2))

def initial_u_func(x, l_domain):
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32, device=DEVICE)
    amplitude = 0.5
    return amplitude * torch.sin(2 * np.pi * x / l_domain)

def initial_rho_star_func(x, l_domain):
    # case 1:
    return 0.7 * torch.ones_like(x)
    # case 2:
    # return 1.0 * torch.ones_like(x)
    # case 3:
    # amplitude    = 0.2
    # background   = 0.6
    # pulse_width  = l_domain / 10.0
    # pulse_center = l_domain / 2.0
    # x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    # return background - amplitude * torch.exp(-((x_tensor - pulse_center) ** 2) /
    #                                           (2 * pulse_width ** 2))

# ============================================================
# Shared collocation sets 
# ============================================================
pde_x = torch.rand(N_PDE, 1, device=DEVICE) * L_DOMAIN
pde_t = torch.rand(N_PDE, 1, device=DEVICE) * T_DOMAIN
pde_collocation_points = torch.cat((pde_x, pde_t), dim=1)

ic_x = torch.rand(N_IC, 1, device=DEVICE) * L_DOMAIN
ic_t = torch.zeros_like(ic_x, device=DEVICE)
ic_collocation_points = torch.cat((ic_x, ic_t), dim=1)

rho_ic_true      = initial_rho_func(ic_x, L_DOMAIN).to(DEVICE)
u_ic_true        = initial_u_func(ic_x, L_DOMAIN).to(DEVICE)
rho_star_ic_true = initial_rho_star_func(ic_x, L_DOMAIN).to(DEVICE)

bc_t  = torch.rand(N_BC, 1, device=DEVICE) * T_DOMAIN
bc_x0 = torch.zeros_like(bc_t, device=DEVICE)
bc_xL = torch.ones_like(bc_t,  device=DEVICE) * L_DOMAIN
bc_coords_left  = torch.cat((bc_x0, bc_t), dim=1)
bc_coords_right = torch.cat((bc_xL, bc_t), dim=1)

# MSE
def mse0(x):
    return x.pow(2).mean()

# ============================================================
# PINN class
# ============================================================
class MLP_PINN(nn.Module):
    """
      - input: (x, t)
      - output (raw): (d_rho, d_u, d_rho_star)
      - final outputs use hard IC conditioning 
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "tanh",
        temperature_rho: float = TEMPERATURE_RHO_DEFAULT,
    ):
        super().__init__()
        self.temperature_rho = temperature_rho

        act_name = activation.lower()
        if act_name == "tanh":
            act = nn.Tanh
        elif act_name == "relu":
            act = nn.ReLU
        elif act_name == "silu":
            act = nn.SiLU
        else:
            raise ValueError("activation must be one of: tanh, relu, silu")

        layers = []
        dims = [input_dim] + list(hidden_layers) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

        # Xavier initialisation
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_input: torch.Tensor):
        """
        x_input: [N, 2] with columns (x, t)
        Returns: rho(x,t), u(x,t), rho_star(x,t)
        """
        raw = self.net(x_input)
        d_rho  = raw[:, 0:1]
        d_u    = raw[:, 1:2]
        d_rs   = raw[:, 2:3]

        x = x_input[:, :1]
        t = x_input[:, 1:2]

        # IC profiles (hard conditioning)
        rho0 = initial_rho_func(x, L_DOMAIN)
        u0   = initial_u_func(x, L_DOMAIN)
        rs0  = initial_rho_star_func(x, L_DOMAIN)

        # scheduler: 0 at t=0, ~1 later
        A = t / T_DOMAIN

        eps = 1e-6
        # keep rho in (0,1), anchored at rho0 at t=0
        rho0_clamped = torch.clamp(rho0, eps, 1.0 - eps)
        rho_val      = torch.sigmoid(torch.logit(rho0_clamped) + A * d_rho)

        # velocity: linear perturbation
        u_val        = u0 + A * d_u

        # rho_star: positive, anchored at rs0
        rho_star_val = torch.clamp(rs0 * torch.exp(A * d_rs), min=STABILITY_EPS)

        return rho_val, u_val, rho_star_val

# ============================================================
# PDE residuals
# ============================================================
def pde_residuals_system_fast(model, xt_coords, mu, gamma, f_ext, stability_eps):
    xt = xt_coords.detach().requires_grad_(True)

    rho, u, rho_star = model(xt)

    grad_rho = torch.autograd.grad(rho, xt, grad_outputs=torch.ones_like(rho),
                                   create_graph=True)[0]
    grad_u   = torch.autograd.grad(u,   xt, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
    grad_rs  = torch.autograd.grad(rho_star, xt,
                                   grad_outputs=torch.ones_like(rho_star),
                                   create_graph=True)[0]

    rho_x, rho_t = grad_rho[:, :1], grad_rho[:, 1:]
    u_x,   u_t   = grad_u[:,   :1], grad_u[:,   1:]
    rs_x,  rs_t  = grad_rs[:,  :1], grad_rs[:,  1:]

    # u_xx
    grad_u_x = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
    u_xx     = grad_u_x[:, :1]

    # (1) Continuity: rho_t + d_x(rho u) = rho_t + (rho_x*u + rho*u_x)
    residual1 = rho_t + rho_x * u + rho * u_x

    # (2) Momentum terms
    dt_rho_u   = rho_t * u + rho * u_t
    dx_rho_u2  = rho_x * (u**2) + 2.0 * rho * u * u_x

    denom = (1.0 - rho + stability_eps)
    a     = mu / denom
    a_x   = mu * rho_x / (denom**2)
    visc  = -(a_x * u_x + a * u_xx)

    rs_safe = rho_star
    q       = rho / rs_safe
    q_x     = (rho_x * rs_safe - rho * rs_x) / (rs_safe**2 + stability_eps)
    dx_p    = gamma * torch.pow(q + stability_eps, gamma - 1.0) * q_x

    force   = rho * f_ext

    residual2 = dt_rho_u + dx_rho_u2 + visc + dx_p - force

    # (3) rho_star_t + u * rho_star_x = 0
    residual3 = rs_t + u * rs_x

    return residual1, residual2, residual3, grad_rs

# ============================================================
# Training + evaluation
# ============================================================
def build_model_from_config(config: Dict[str, Any]) -> MLP_PINN:
    hidden_layers = config.get("hidden_layers", [128, 128, 128, 128])
    activation    = config.get("activation", "tanh")
    temp_rho      = config.get("temperature_rho", TEMPERATURE_RHO_DEFAULT)
    model = MLP_PINN(
        input_dim=2,
        hidden_layers=hidden_layers,
        output_dim=3,
        activation=activation,
        temperature_rho=temp_rho,
    ).to(DEVICE)
    return model

def train_and_evaluate(config: Dict[str, Any]):
    name   = config['name']
    gamma  = config.get('gamma', GAMMA_INDEX)
    mu     = config.get('mu', MU_VISCOSITY)
    f_ext  = config.get('f_ext', FORCE_F_CONST)
    eps    = config.get('stability_eps', STABILITY_EPS)

    # loss weights
    lam_pde1 = config.get('lambda_pde1', LAMBDA_PDE1)
    lam_pde2 = config.get('lambda_pde2', LAMBDA_PDE2)
    lam_pde3 = config.get('lambda_pde3', LAMBDA_PDE3)
    lam_ic   = config.get('lambda_ic',  LAMBDA_IC)
    lam_bc   = config.get('lambda_bc',  LAMBDA_BC)
    lam_dx_rho_ic     = config.get('lambda_dx_rho_ic',     LAMBDA_DX_RHO_IC)
    lam_dx_rhostar_ic = config.get('lambda_dx_rhostar_ic', LAMBDA_DX_RHOSTAR_IC)

    # training params
    lr_adam   = config.get('learning_rate_adam',  LEARNING_RATE_ADAM)
    lr_lbfgs  = config.get('learning_rate_lbfgs', LEARNING_RATE_LBFGS)
    adam_ep   = config.get('adam_epochs',  ADAM_EPOCHS_DEFAULT)
    lbfgs_it  = config.get('lbfgs_steps',  LBFGS_STEPS_DEFAULT)
    seed      = config.get('random_seed', 42)

    torch.manual_seed(seed); np.random.seed(seed)

    model = build_model_from_config(config)

    print(f"---[{name}] γ={gamma}, μ={mu}, Adam={adam_ep} ep, L-BFGS={lbfgs_it} steps ---")

    # closure for computing loss (for both Adam and L-BFGS)
    def compute_loss():
        # PDE residuals
        r1, r2, r3, r4 = pde_residuals_system_fast(model, pde_collocation_points,
                                               mu, gamma, f_ext, eps)
        loss_pde = lam_pde1 * mse0(r1) + lam_pde2 * mse0(r2) + lam_pde3 * mse0(r3) 
        
        if IC_RS_CONST:
            loss_pde += lam_pde3 * mse0(r4)

        # IC
        rho_i, u_i, rs_i = model(ic_collocation_points)
        loss_ic = lam_ic * (
            mse0(rho_i - rho_ic_true) +
            mse0(u_i   - u_ic_true) +
            mse0(rs_i  - rho_star_ic_true)
        )

        # BC 
        rho_l, u_l, rs_l = model(bc_coords_left)
        rho_r, u_r, rs_r = model(bc_coords_right)
        loss_bc = lam_bc * (
            mse0(rho_l - rho_r) +
            mse0(u_l   - u_r) +
            mse0(rs_l  - rs_r)
        )

        # IC derivative penalties (can turn off)
        loss_dx_ic = torch.zeros((), device=DEVICE)
        if lam_dx_rho_ic > 0 or lam_dx_rhostar_ic > 0:
            ic_pts_grad = ic_collocation_points.detach().requires_grad_(True)
            rho_g, _, rs_g = model(ic_pts_grad)
            if lam_dx_rho_ic > 0:
                dx_rho_ic = torch.autograd.grad(rho_g, ic_pts_grad,
                                                grad_outputs=torch.ones_like(rho_g),
                                                create_graph=True)[0][:, :1]
                loss_dx_ic = loss_dx_ic + lam_dx_rho_ic * mse0(dx_rho_ic)
            if lam_dx_rhostar_ic > 0:
                dx_rs_ic = torch.autograd.grad(rs_g, ic_pts_grad,
                                               grad_outputs=torch.ones_like(rs_g),
                                               create_graph=True)[0][:, :1]
                loss_dx_ic = loss_dx_ic + lam_dx_rhostar_ic * mse0(dx_rs_ic)

        total_loss = loss_pde + loss_ic + loss_bc + loss_dx_ic
        return total_loss, loss_pde, loss_ic, loss_bc, loss_dx_ic

    # -------------------- Adam phase --------------------
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    t0 = time.time()
    model.train()
    log_every = max(adam_ep // 5, 1)

    for ep in range(1, adam_ep + 1):
        optimizer_adam.zero_grad()
        loss, lpde, lic, lbc, ldx = compute_loss()
        loss.backward()
        optimizer_adam.step()

        if ep % log_every == 0 or ep == 1 or ep == adam_ep:
            print(f"[{name}] Adam ep {ep:5d}/{adam_ep} | "
                  f"loss {loss.item():.3e} | PDE {lpde.item():.2e} | "
                  f"IC {lic.item():.2e} | BC {lbc.item():.2e} | dIC {ldx.item():.2e}")

    # -------------------- L-BFGS phase --------------------
    print(f"[{name}] switching to L-BFGS...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lr_lbfgs,
        max_iter=lbfgs_it,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    iter_count = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, lpde, lic, lbc, ldx = compute_loss()
        loss.backward()
        iter_count[0] += 1
        if iter_count[0] % max(lbfgs_it // 5, 1) == 0 or iter_count[0] == 1:
            print(f"[{name}] L-BFGS iter {iter_count[0]:5d}/{lbfgs_it} | "
                  f"loss {loss.item():.3e} | PDE {lpde.item():.2e} | "
                  f"IC {lic.item():.2e} | BC {lbc.item():.2e} | dIC {ldx.item():.2e}")
        return loss

    model.train()
    optimizer_lbfgs.step(closure)

    print(f"[{name}] training done in {time.time() - t0:.1f}s\n")

    # -------------------- Evaluation grid --------------------
    model.eval()
    with torch.no_grad():
        xg = torch.linspace(0, L_DOMAIN, 100, device=DEVICE)
        tg = torch.linspace(0, T_DOMAIN, 60,  device=DEVICE)
        Xg, Tg = torch.meshgrid(xg, tg, indexing='ij')
        pts = torch.stack((Xg.flatten(), Tg.flatten()), dim=1)
        r_f, u_f, rs_f = model(pts)

    return {
        'name':     name,
        'X':        Xg.cpu().numpy(),
        'T':        Tg.cpu().numpy(),
        'Rho':      r_f.reshape(100, 60).cpu().numpy(),
        'U':        u_f.reshape(100, 60).cpu().numpy(),
        'Rho_star': rs_f.reshape(100, 60).cpu().numpy()
    }

# ============================================================
# Configs
# ============================================================
configs = [
    {
        'name': 'G=2',
        'gamma': 2.0,
        'mu': 1e-3,
        'hidden_layers': [256, 256, 256, 256],
        'activation': 'tanh',
        'adam_epochs': 5500,
        'lbfgs_steps': 2500,
        'lambda_ic': 1.0,
        'lambda_bc': 1.0,
        'lambda_dx_rho_ic': 0.0,
        'lambda_dx_rhostar_ic': 0.0,
    },
    {
        'name': 'G=5',
        'gamma': 5.0,
        'mu': 1e-3,
        'hidden_layers': [256, 256, 256, 256],
        'activation': 'tanh',
        'adam_epochs': 5500,
        'lbfgs_steps': 2500,
        'lambda_ic': 1.0,
        'lambda_bc': 1.0,
    },
    {
        'name': 'G=10',
        'gamma': 10.0,
        'mu': 1e-3,
        'hidden_layers': [256, 256, 256, 256, 256],
        'activation': 'tanh',
        'adam_epochs': 5500,
        'lbfgs_steps': 2500,
        'lambda_ic': 1.0,
        'lambda_bc': 1.0,
    }
]

# ============================================================
# Train 
# ============================================================
results = [train_and_evaluate(cfg) for cfg in configs]

# ============================================================
# Plotting 
# ===========================================================
target_times = [0.0, 0.075, 0.15, 0.225, 0.3]
fields_profile = ['Rho', 'U', 'Rho_star']
titles_profile = [r'$\rho(x,t)$', r'$u(x,t)$', r'$\rho^\star(x,t)$']

# build 1D time grid from first result
if results[0]['T'].ndim == 2:
    t_grid = results[0]['T'][0, :]
else:
    t_grid = results[0]['T']

t_min = float(t_grid[0]); t_max = float(t_grid[-1])
nt = int(t_grid.shape[0])
dt = (t_max - t_min) / max(nt - 1, 1)

def time_to_index(t):
    t_clamped = min(max(float(t), t_min), t_max)
    idx = int(round((t_clamped - t_min) / dt))
    return max(0, min(nt - 1, idx))

time_indices = [time_to_index(t) for t in target_times]
actual_times = [float(t_grid[idx]) for idx in time_indices]

fig, axs = plt.subplots(len(fields_profile), len(time_indices),
                        figsize=(16, 10), sharex=True)

for r, fld in enumerate(fields_profile):
    for c, (t_req, tidx) in enumerate(zip(target_times, time_indices)):
        ax = axs[r, c] if len(fields_profile) > 1 else axs[c]
        for res in results:
            x_line = res['X'][:, tidx] if res['X'].ndim == 2 else res['X']
            y_line = res[fld][:, tidx]
            ax.plot(x_line, y_line, label=res['name'])

        ax.set_title(f'{titles_profile[r]} at t ≈ {actual_times[c]:.3f}')
        ax.grid(True)

        if fld == 'U':
            ax.set_ylim(-1.0, 1.0)
        else:
            ax.set_ylim(0.0, 1.1)

        if r == len(fields_profile) - 1:
            ax.set_xlabel('x')
        if c == 0:
            ax.set_ylabel(titles_profile[r])
            ax.legend()

plt.tight_layout()
plt.show()