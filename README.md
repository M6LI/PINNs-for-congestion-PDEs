# PINNs for a Congestion-Driven PDE with Transported Capacity
Based on arXiv:[TBC]
This repository contains a **Physics-Informed Neural Network (PINN)** implementation in PyTorch for a congestion-type PDE system, which is rigorously derived from particle dynamics in the above paper. The aim of the code is to build a network which helps to improve our understanding of the qualitative properties of this system without resorting to traditional methods (e.g. finite difference); these do not scale well to higher dimensions.

The implementation uses a **two-stage optimisation strategy (Adam → L-BFGS)** and enforces initial conditions through **hard conditioning** built directly into the neural network architecture.

---

## PDE model

The model solved by the PINN is the 1D system
The PDE system is given by:

- **Continuity equation**
  
  $\partial_t \rho + \partial_x(\rho u) = 0$

- **Momentum equation**
  
  `∂_t(ρ u) + ∂_x(ρ u²)
   − ∂_x( (μ / (1 − ρ)) ∂_x u )
   + ∂_x( (ρ / ρ*)^γ )
   = ρ f`

- **Transport of congestion threshold**
  
  `∂_t ρ* + u ∂_x ρ* = 0`



### Variables and parameters
- $\rho(x,t)$ : density  
- $u(x,t)$ : velocity  
- $\rho^\*(x,t)$ : transported congestion threshold (capacity)  
- $\mu > 0$ : viscosity parameter (singular as $\rho \to 1$)  
- $\gamma > 0$ : congestion exponent  
- $f$ : external forcing (constant in this code)

This system combines:
- hyperbolic transport,
- singular / degenerate viscosity,
- a **moving congestion constraint** through $\rho^\*$.

## Network architecture

The PINN is a fully connected multilayer perceptron (MLP):

- **Input:** $(x,t)$  
- **Output (raw):** $(\delta \rho,\ \delta u, \ \delta \rho^\*)$  
- **Hidden layers:** configurable (default: 5 layers of width 256)  
- **Activation:** `tanh` (configurable)

### Hard initial-condition conditioning

The final network outputs are constructed as
- $\rho(t,x) = \rho_0(x)\\exp\big(A(t)\delta\rho\big)$  
- $u(t,x) = u_0(x) + A(t)\ u$  
- $\rho^\*(t,x) = \rho^\*_0(x)\,\exp\big(A(t)\\delta\rho^\*\big)$  

where $A(t)=t/T$ is a time-scheduling factor that ensures **exact satisfaction of the initial conditions at $t=0$**.

---

## Loss function

The total loss is given by
$$
\mathcal{L}
= \mathcal{L}_{\text{PDE}}
+ \mathcal{L}_{\text{IC}}
+ \mathcal{L}_{\text{BC}}
+ \mathcal{L}_{\text{IC-derivatives}}.
$$

Here:
- $\mathcal{L}_{\text{PDE}}$ enforces the three PDE equations,
- $\mathcal{L}_{\text{IC}}$ enforces the initial data,
- $\mathcal{L}_{\text{BC}}$ enforces boundary consistency,
- $\mathcal{L}_{\text{IC-derivatives}}$ (optional) penalises spatial derivatives at $t=0$ to regularise initial gradients.

All loss weights are fully configurable.

---





