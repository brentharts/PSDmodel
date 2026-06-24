#!/usr/bin/env python3
"""
kramers_escape_sim.py
=====================
Numerical verification of the AMPAR-retention law used in the PSD-condensate
model. The paper writes receptor residence time as

        tau_res = tau0 * exp(dG / kB T)            (Kramers, high-barrier limit)

Here we (1) simulate an overdamped Brownian receptor escaping a trap, (2)
compare the measured mean escape time to the EXACT 1-D mean-first-passage-time
(MFPT) integral, and (3) show that the simple exponential is the correct
large-barrier asymptote of that integral.

Comparing to the exact MFPT (not just the exponential) is the honest test: the
pure exponential has slope 1 only asymptotically, because curvature prefactors
add slowly-varying corrections over a finite barrier range. The simulation
should match the exact integral across the whole range.

This is a MODEL self-consistency check. It uses no experimental data and is not
evidence for the biological hypothesis -- only that the escape-kinetics math in
the paper is internally correct.
"""

import numpy as np
import matplotlib.pyplot as plt


def U(x, barrier, L):
    return barrier * (1.0 - np.cos(np.pi * x / L)) / 2.0

def dU(x, barrier, L):
    return barrier * (np.pi / (2.0 * L)) * np.sin(np.pi * x / L)


def simulate_escape_times(barrier, n_traj=1500, L=1.0, D=1.0,
                          dt=2e-4, x0=1e-3, max_steps=40_000_000, seed=0):
    """Overdamped Langevin: dx = -D U'(x) dt + sqrt(2 D dt) xi.
    Reflecting wall at x=0, absorbing boundary at x=L."""
    rng = np.random.default_rng(seed)
    x = np.full(n_traj, x0, dtype=float)
    escaped = np.zeros(n_traj, dtype=bool)
    t_escape = np.zeros(n_traj, dtype=float)
    amp = np.sqrt(2.0 * D * dt)
    steps = 0
    while not escaped.all() and steps < max_steps:
        live = ~escaped
        n = live.sum()
        xi = rng.standard_normal(n)
        x[live] = np.abs(x[live] - D * dU(x[live], barrier, L) * dt + amp * xi)
        idx = np.where(live)[0]
        newly = idx[x[live] >= L]
        if newly.size:
            escaped[newly] = True
            t_escape[newly] = (steps + 1) * dt
        steps += 1
    if not escaped.all():
        t_escape[~escaped] = steps * dt
    return t_escape, bool(escaped.all())


def mfpt_exact(barrier, L=1.0, D=1.0, x0=0.0, n=4000):
    """Exact 1-D MFPT, reflecting at 0, absorbing at L:
        tau(x0) = (1/D) int_{x0}^{L} dy e^{U(y)} int_0^{y} e^{-U(z)} dz ."""
    x = np.linspace(0.0, L, n)
    Ux = U(x, barrier, L)
    inner = np.concatenate([[0.0],
              np.cumsum(0.5 * (np.exp(-Ux[1:]) + np.exp(-Ux[:-1])) * np.diff(x))])
    integrand = np.exp(Ux) * inner
    i0 = np.searchsorted(x, x0)
    return np.trapezoid(integrand[i0:], x[i0:]) / D


def main():
    barriers = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    sim, exact = [], []
    for b in barriers:
        t_esc, done = simulate_escape_times(b, seed=int(b * 13))
        sim.append(t_esc.mean())
        exact.append(mfpt_exact(b))
        print(f"barrier={b:.1f} kT   sim=<{t_esc.mean():.4g}>   "
              f"exact_MFPT={exact[-1]:.4g}   ratio={t_esc.mean()/exact[-1]:.3f}   "
              f"all_escaped={done}")
    sim, exact = np.array(sim), np.array(exact)

    bb = np.linspace(barriers.min() - 0.3, barriers.max() + 0.3, 200)
    norm = exact[-1] / np.exp(barriers[-1])
    asymptote = norm * np.exp(bb)
    exact_curve = np.array([mfpt_exact(b) for b in bb])

    fig, ax = plt.subplots(figsize=(6.7, 4.7))
    ax.semilogy(bb, exact_curve, "-", color="#aa2323", lw=2,
                label="exact Kramers MFPT integral")
    ax.semilogy(bb, asymptote, ":", color="#197846", lw=1.8,
                label=r"$\propto e^{\Delta G/k_BT}$ (large-barrier asymptote)")
    ax.semilogy(barriers, sim, "o", ms=8, color="#1a5096",
                label="Langevin simulation", zorder=5)
    ax.set_xlabel(r"barrier height  $\Delta G / k_BT$")
    ax.set_ylabel(r"mean escape time  $\langle\tau\rangle$  (units of $L^2/D$)")
    ax.set_title("Receptor escape: simulation vs. exact Kramers theory")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig("kramers_verification.pdf")
    fig.savefig("kramers_verification.png", dpi=140)

    rms = np.sqrt(np.mean((sim / exact - 1.0) ** 2))
    print(f"\nRMS fractional deviation sim vs exact: {rms*100:.1f}%")
    print("saved kramers_verification.pdf / .png")


if __name__ == "__main__":
    main()
