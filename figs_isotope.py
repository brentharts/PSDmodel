import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":140,"font.family":"DejaVu Sans"})
C_DENSE="#1a5096"; C_GREEN="#197846"; C_RED="#aa2323"; C_DIL="#be820a"

p_c=1/3.0; beta=0.41; p0=0.50; g=0.12   # connectivity gain per unit D2O mole fraction
def Pinf(p): return np.where(p>p_c, ((p-p_c)/(1-p_c))**beta, 0.0)

xD=np.linspace(0,1,400)
p=p0+g*xD
fig,ax=plt.subplots(figsize=(6.4,4.5))
for dG,ls in [(8,":"),(12,"-"),(15,"--")]:
    fold=np.exp(dG*(Pinf(p)-Pinf(np.array(p0))))
    ax.plot(xD,fold,ls,color=C_DENSE,label=r"$\Delta G_{\max}=%d\,k_BT$"%dG)
ax.axhline(1,color="gray",lw=0.8,ls=":")
# physiological partial-substitution band
ax.axvspan(0,0.4,color=C_GREEN,alpha=0.08)
ax.text(0.2,ax.get_ylim()[1]*0.92,"partial substitution\n(experimentally accessible)",
        ha="center",fontsize=8.5,color=C_GREEN)
# DDW annotation
ax.annotate("DDW (depletion):\npredicted small\n$\\tau_{res}$ decrease",
            xy=(0.0,1.0),xytext=(0.04,0.55),fontsize=8.5,color=C_DIL,
            arrowprops=dict(arrowstyle="-|>",color=C_DIL,lw=1.4))
ax.set_xlabel(r"D$_2$O mole fraction  $x_D$")
ax.set_ylabel(r"retention fold-change  $\tau_{res}(x_D)/\tau_{res}(0)$")
ax.set_title("Predicted isotope effect on AMPAR retention (speculative)",fontsize=11)
ax.legend(frameon=False,fontsize=9,loc="center right")
plt.tight_layout(); plt.savefig("fig_isotope.pdf"); plt.close()

# worked number for the text
for xv in [0.5,1.0]:
    pv=p0+g*xv; print("x_D=%.2f  p=%.3f  Pinf=%.3f  fold(12kT)=%.2f"%(
        xv,pv,Pinf(np.array(pv)),np.exp(12*(Pinf(np.array(pv))-Pinf(np.array(p0))))))
print("done")
