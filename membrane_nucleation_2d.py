#!/usr/bin/env python3
"""
membrane_nucleation_2d.py
=========================
Two-dimensional nucleation model of AMPAR nanodomains -- the honest refinement
of the 3-D capillary estimate (R* = 2 gamma / Delta f) used in the paper.

The postsynaptic membrane is two-dimensional, so a forming receptor nanodomain
is a 2-D disk, not a 3-D droplet. Its free energy trades a line (perimeter)
cost against an areal condensation gain:

    Delta G(R) = - pi R^2 * df2D  +  2 pi R * lambda

  lambda : line tension of the domain edge            [N]  (= J/m)
  df2D   : areal free-energy density of condensation  [J/m^2]

Critical radius and barrier (classical 2-D nucleation):

    R*      = lambda / df2D
    DeltaG* = pi lambda^2 / df2D
    N*      = pi R*^2 * rho_s        (receptors per critical domain)

This script computes R*, the nucleation barrier, and N*, and compares them to
super-resolution measurements (~70-80 nm domains, ~20 receptors). A substantial
barrier means nanodomain formation is activity-GATED (must be nucleated, e.g. by
CaMKII recruitment), which is consistent with LTP being an induced event.

MODEL exploration only -- not a fit to data, not evidence for the hypothesis.
"""

import numpy as np
import matplotlib.pyplot as plt

kT = 4.28e-21          # J at 310 K
C1, C2, C3 = "#1a5096", "#aa2323", "#197846"


def df2D(eps_kT, rho_s):
    """Areal condensation free-energy density from per-receptor binding gain
    eps (in kT) and areal density rho_s (1/m^2)."""
    return eps_kT * kT * rho_s


def R_star(lmbda, df):           # critical radius [m]
    return lmbda / df

def barrier_kT(lmbda, df):       # nucleation barrier [kT]
    return np.pi * lmbda**2 / df / kT

def N_star(lmbda, df, rho_s):    # receptors in critical domain
    return np.pi * R_star(lmbda, df)**2 * rho_s


def main():
    rho_s = 2e15            # receptors+scaffold sites per m^2 (~2000 / um^2)
    eps_kT = 3.0            # per-receptor condensation gain (kT)
    df = df2D(eps_kT, rho_s)
    print(f"areal free-energy density df2D = {df:.3e} J/m^2")

    lam = np.logspace(-13, -11.3, 300)   # line tension 0.1 - ~5 pN

    # ---- Panel 1: critical radius vs line tension ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for e, ls in [(2.0, ":"), (3.0, "-"), (5.0, "--")]:
        d = df2D(e, rho_s)
        ax.plot(lam * 1e12, 2 * R_star(lam, d) * 1e9, ls, color=C1,
                label=fr"$\epsilon={e:.0f}\,k_BT$")   # diameter = 2R*
    ax.axhspan(70, 80, color=C3, alpha=0.20)
    ax.text(0.12, 75, "measured\n~70-80 nm", color=C3, fontsize=9, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"line tension  $\lambda$  (pN)")
    ax.set_ylabel(r"critical domain diameter  $2R^*$  (nm)")
    ax.set_title("2-D nanodomain size")
    ax.legend(frameon=False, fontsize=9)

    # ---- Panel 2: nucleation barrier vs line tension ----
    ax = axes[1]
    for e, ls in [(2.0, ":"), (3.0, "-"), (5.0, "--")]:
        d = df2D(e, rho_s)
        ax.plot(lam * 1e12, barrier_kT(lam, d), ls, color=C2,
                label=fr"$\epsilon={e:.0f}\,k_BT$")
    ax.axhspan(0, 5, color="gray", alpha=0.12)
    ax.text(0.12, 2.5, "near-spontaneous\n($\\Delta G^*\\lesssim 5\\,k_BT$)",
            fontsize=8.5, va="center", color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"line tension  $\lambda$  (pN)")
    ax.set_ylabel(r"nucleation barrier  $\Delta G^*/k_BT$")
    ax.set_title("Barrier: domains are activity-gated")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig("membrane_nucleation_2d.pdf")
    fig.savefig("membrane_nucleation_2d.png", dpi=140)

    # ---- worked point ----
    lam0 = 1e-12
    print(f"\nAt lambda=1 pN, eps=3 kT, rho_s=2000/um^2:")
    print(f"  R*       = {R_star(lam0, df)*1e9:5.1f} nm  (diameter {2*R_star(lam0,df)*1e9:.1f} nm)")
    print(f"  DeltaG*  = {barrier_kT(lam0, df):5.1f} kT   -> must be nucleated")
    print(f"  N*       = {N_star(lam0, df, rho_s):5.1f} receptors (measured ~20)")
    print("saved membrane_nucleation_2d.pdf / .png")


if __name__ == "__main__":
    main()
