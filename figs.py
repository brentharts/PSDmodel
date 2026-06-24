"""
Quantitative figures for the PSD-condensate / AMPAR-retention modeling paper.
All physics is standard soft-matter applied to the synaptic question.
"""
import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11, "axes.linewidth": 0.9, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})

C_DENSE = "#1a5096"; C_DILUTE = "#be820a"; C_RED = "#aa2323"
C_GREEN = "#197846"; C_TEAL = "#0a8c78"

# =====================================================================
# FIGURE 1 : Flory-Huggins phase diagram (binodal + spinodal) + percolation line
# =====================================================================
# f(phi)/kT = (phi/N) ln phi + (1-phi) ln(1-phi) + chi phi (1-phi)
def N_poly(): return 25.0   # effective degree of polymerization / valency proxy

def df(phi, N):       # f'(phi)
    return (np.log(phi)+1)/N - (np.log(1-phi)+1) + 1*(1-2*phi)*0  # chi term added separately
# We need f' including chi; write full:
def fprime(phi, chi, N):
    return (np.log(phi)+1.0)/N - (np.log(1-phi)+1.0) + chi*(1.0-2.0*phi)
def f_of(phi, chi, N):
    return (phi/N)*np.log(phi) + (1-phi)*np.log(1-phi) + chi*phi*(1-phi)

def spinodal_chi(phi, N):       # f''=0  -> chi = 0.5*(1/(N phi) + 1/(1-phi))
    return 0.5*(1.0/(N*phi) + 1.0/(1.0-phi))

N = N_poly()
phi_c = 1.0/(1.0+np.sqrt(N))
chi_c = spinodal_chi(phi_c, N)

# spinodal curve
phi_sp = np.linspace(1e-4, 1-1e-4, 4000)
chi_sp = spinodal_chi(phi_sp, N)

# binodal: for each chi>chi_c solve common-tangent (equal mu and equal osmotic pressure)
def binodal_pair(chi, N):
    # two unknowns phi_a<phi_c<phi_b ; equations: fprime equal, and grand-potential equal
    def eqs(x):
        a, b = x
        if a<=0 or a>=1 or b<=0 or b>=1: return [1e3,1e3]
        e1 = fprime(a, chi, N) - fprime(b, chi, N)
        e2 = (f_of(a,chi,N)-a*fprime(a,chi,N)) - (f_of(b,chi,N)-b*fprime(b,chi,N))
        return [e1, e2]
    # initial guesses bracketing the critical point
    a0 = max(phi_c*0.2, 1e-3); b0 = min(phi_c + (1-phi_c)*0.6, 1-1e-3)
    sol, info, ier, msg = fsolve(eqs, [a0, b0], full_output=True)
    if ier==1 and sol[0]>0 and sol[1]<1 and abs(sol[0]-sol[1])>1e-3:
        return sorted(sol)
    return None

chis = np.linspace(chi_c*1.0001, chi_c*2.4, 80)
bino_lo=[]; bino_hi=[]; bino_chi=[]
for ch in chis:
    pr = binodal_pair(ch, N)
    if pr:
        bino_lo.append(pr[0]); bino_hi.append(pr[1]); bino_chi.append(ch)
bino_lo=np.array(bino_lo); bino_hi=np.array(bino_hi); bino_chi=np.array(bino_chi)

# Flory-Stockmayer percolation/gel line (illustrative): gelation when chi exceeds chi_gel(phi)
# Use a simple connectivity proxy p = phi * z_eff; percolation at p_c = 1/(f-1).
f_func = 4.0  # scaffold functionality (e.g. PSD-95 / SynGAP / Shank multivalency)
p_c = 1.0/(f_func-1.0)

fig, ax = plt.subplots(figsize=(6.4,4.7))
ax.plot(bino_lo, bino_chi, color=C_DENSE, lw=2.3)
ax.plot(bino_hi, bino_chi, color=C_DENSE, lw=2.3, label="Binodal (coexistence)")
ax.plot(phi_sp, chi_sp, color=C_RED, lw=1.8, ls="--", label="Spinodal (f''=0)")
ax.plot([phi_c],[chi_c],"o", color="k", ms=6, zorder=5)
ax.annotate("critical point\n$(\\phi_c,\\chi_c)$", (phi_c,chi_c),
            textcoords="offset points", xytext=(12,6), fontsize=9)

# shade dense (gel/percolated) region above an illustrative percolation threshold in the dense branch
ax.fill_betweenx(bino_chi, bino_hi, 1.0, color=C_DENSE, alpha=0.08)
ax.text(0.86, chi_c*2.0, "dense phase\n(percolated\nsoft glass)",
        fontsize=8.5, color=C_DENSE, ha="center")
ax.text(0.13, chi_c*2.0, "dilute phase\n($c_{sat}$ branch)",
        fontsize=8.5, color=C_DILUTE, ha="center")

# arrow showing CaMKII-GluN2B / Shank3 oligomerization raising effective chi
ax.annotate("", xy=(0.5, chi_c*2.15), xytext=(0.5, chi_c*1.15),
            arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2))
ax.text(0.52, chi_c*1.62, "CaMKII–GluN2B binding,\nShank3 oligomerization\n$\\Rightarrow$ effective $\\chi\\uparrow$",
        fontsize=8.5, color=C_GREEN)

ax.set_xlabel("scaffold volume fraction  $\\phi$")
ax.set_ylabel("effective interaction parameter  $\\chi$")
ax.set_xlim(0,1); ax.set_ylim(chi_c*0.85, chi_c*2.45)
ax.set_title("PSD phase behaviour (Flory–Huggins baseline, $N=%d$)"%int(N), fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
plt.tight_layout(); plt.savefig("fig_phase.pdf"); plt.close()
print("phi_c=%.4f chi_c=%.4f p_c=%.3f"%(phi_c,chi_c,p_c))

# =====================================================================
# FIGURE 2 : Kramers / partitioning retention time vs network connectivity
#   tau_res = tau0 * exp(dG_eff/kT),  dG_eff = dG_max * P_inf(p)
#   P_inf(p) = ((p-p_c)/(1-p_c))^beta  for p>p_c, else 0   (percolation order param)
# =====================================================================
kT = 4.28e-21          # J at 310 K
tau0 = 1e-3            # s, microscopic attempt/encounter time
dG_max = 12.0*kT       # maximal partition free energy at full percolation (~12 kT)
beta = 0.41            # 3D percolation order-parameter exponent

p = np.linspace(0.0, 1.0, 600)
Pinf = np.where(p>p_c, ((p-p_c)/(1-p_c))**beta, 0.0)
dG = dG_max*Pinf
tau = tau0*np.exp(dG/kT)

fig, ax = plt.subplots(figsize=(6.4,4.5))
ax.semilogy(p, tau, color=C_DENSE, lw=2.4)
ax.axvline(p_c, color=C_RED, ls="--", lw=1.6)
ax.text(p_c+0.01, tau[ -1]*0.4, "percolation\nthreshold $p_c$", color=C_RED, fontsize=9)

# mark healthy (potentiated) and perturbed (hexanediol / Shank3 monomer) states
p_health = 0.62; p_pert = 0.30
for pv,lab,col,dy in [(p_health,"LTP-maintained\n(percolated PSD)",C_GREEN,1.6),
                      (p_pert,"LLPS disrupted\n(hexanediol / Shank3\nmonomer)",C_DILUTE,3)]:
    tv = tau0*np.exp(dG_max*(((pv-p_c)/(1-p_c))**beta if pv>p_c else 0.0)/kT)
    ax.plot([pv],[tv],"o",color=col,ms=7,zorder=5)
    ax.annotate(lab,(pv,tv),textcoords="offset points",xytext=(8,-2 if col==C_DILUTE else 6),
                fontsize=8.5,color=col)

ax.set_xlabel("scaffold network connectivity  $p$  (bond occupancy)")
ax.set_ylabel("AMPAR residence time  $\\tau_{res}$  (s)")
ax.set_title("Receptor retention as Kramers escape from the condensate cage", fontsize=11)
ax.set_xlim(0,1)
ax.grid(True, which="both", axis="y", alpha=0.18)
plt.tight_layout(); plt.savefig("fig_retention.pdf"); plt.close()
print("tau healthy ~ %.2g s ; tau perturbed ~ %.2g s"%(
      tau0*np.exp(dG_max*(((p_health-p_c)/(1-p_c))**beta)/kT), tau0))

# =====================================================================
# FIGURE 3 : nucleation / capillarity cluster size  R* = 2 gamma / Delta f
#   3D estimate + measured AMPAR nanodomain band (~70-80 nm)
# =====================================================================
gamma = np.logspace(-7, -3, 300)     # surface tension N/m (0.1 - 1000 uN/m)
# supersaturation free-energy density Delta f = kT * c_bind ; c_bind ~ 100 uM .. 2 mM
for c_uM, ls in [(100,"-"),(500,"--"),(2000,":")]:
    c_num = c_uM*1e-6*1e3*6.022e23     # molecules / m^3  (uM -> mol/m3 -> /m3)
    Df = kT*c_num
    Rstar = 2*gamma/Df
    plt.plot(gamma*1e6, Rstar*1e9, ls, color=C_DENSE,
             label="$c_{bind}=%d\\,\\mu$M"%c_uM)
plt.axhspan(70,80, color=C_GREEN, alpha=0.20)
plt.text(0.13, 75, "measured AMPAR\nnanodomain (~70–80 nm)", color=C_GREEN, fontsize=8.5, va="center")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("condensate surface tension  $\\gamma$  ($\\mu$N/m)")
plt.ylabel("critical domain radius  $R^*$  (nm)")
plt.title("Capillary cluster-size estimate vs. measured nanodomain", fontsize=11)
plt.legend(frameon=False, fontsize=9)
plt.ylim(1,1e4)
plt.tight_layout(); plt.savefig("fig_nucleation.pdf"); plt.close()

# print an order-of-magnitude check at gamma=10 uN/m, c=500 uM
g0=10e-6; c0=500e-6*1e3*6.022e23; Df0=kT*c0; R0=2*g0/Df0
print("R* at gamma=10uN/m, c=500uM : %.1f nm"%(R0*1e9))
print("done")
