# PINNs for a Congestion-Driven PDE with Transported Capacity

This repository contains a **Physics-Informed Neural Network (PINN)** implementation (PyTorch) for a congestion-type PDE system derived in our work.  
The aim is to explore and approximate solutions in regimes close to **hard congestion**, where classical numerical methods and standard PINNs face significant challenges.

---

## PDE model

We consider the following 1D system:
$
\begin{cases}
\partial_t \rho + \partial_x(\rho u) = 0, \\\\
\partial_t(\rho u) + \partial_x(\rho u^2)
- \partial_x\!\left( \dfrac{\mu}{1-\rho}\,\partial_x u \right)
+ \partial_x\!\left( \left(\dfrac{\rho}{\rho^\*}\right)^\gamma \right)
= \rho f, \\\\
\partial_t \rho^\* + u\,\partial_x \rho^\* = 0.
\end{cases}
$

**Variables and parameters**
- \(\rho(t,x)\): density  
- \(u(t,x)\): velocity  
- \(\rho^\*(t,x)\): transported congestion threshold (capacity)  
- \(\mu > 0\): viscosity parameter (note the singular factor \(1/(1-\rho)\))  
- \(\gamma > 0\): congestion exponent  
- \(f(t,x)\): forcing term  

This system combines:
- hyperbolic transport,
- singular/degenerate viscosity near \(\rho \to 1\),
- a **transported unilateral constraint** via \(\rho^\*\).

As \(\gamma \to \infty\) (or \(\mu \to 0\)), solutions approach a hard-congestion regime where \(\rho\) behaves like an indicator function.

---

## Purpose of this repository

- Implement a PINN for a **nonstandard congestion PDE** derived analytically.
- Study how PINNs behave near **singular and nonsmooth limits**.
- Highlight strengths and limitations of smooth neural approximations in hard-congestion regimes.

> **Important:**  
> This code is not intended as a solver of record for the hard-congestion limit.  
> It is meant as a *scientific ML experiment* probing where and why PINNs succeed or fail.

---

## Repository structure (example)

