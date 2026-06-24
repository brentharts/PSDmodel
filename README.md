# PSD-Condensate Model — Companion Code

Python scripts accompanying the paper *Material Properties of the Postsynaptic
Density Condensate Set AMPA-Receptor Retention Time: A Quantitative
Phase-Separation and Escape-Kinetics Model of LTP Maintenance* (B. S.
Hartshorn).

These scripts implement and explore the equations in that paper using NumPy and
Matplotlib.

---

## What this code is — and is not

**It is** a set of transparent, runnable implementations of the model's
mathematics: you can change parameters and see what the framework predicts, and
one script independently checks that the analytic retention law is the correct
asymptotics of an explicit stochastic dynamics.

**It is not** experimental validation. None of these scripts use experimental
data, and none of them constitute evidence that the biological hypothesis is
*true*. They show that the model is **internally consistent** and **quantitative**
— i.e., that its predictions are well-defined and fall in physically sensible
ranges. Whether the hypothesis is correct is an empirical question to be settled
by the experiments proposed in the paper, not by these plots. Please cite and
describe the repo accordingly.

---

## Scripts

### `kramers_escape_sim.py` — retention law, verified
Simulates an overdamped Brownian receptor escaping a trap (Langevin dynamics)
and compares the measured mean escape time to the **exact** 1-D mean-first-passage
-time integral, then shows that the paper's simple `tau_res = tau0 * exp(dG/kBT)`
is the correct large-barrier asymptote. The simulation matches the exact theory
to ~5% (the residual is the expected finite-timestep bias). This is a
self-consistency check on the escape-kinetics math.

Output: `kramers_verification.pdf/.png`

### `membrane_nucleation_2d.py` — nanodomain size, refined
A **two-dimensional** nucleation model of AMPAR nanodomains — the honest
refinement of the 3-D capillary estimate in the paper, since the postsynaptic
membrane is 2-D. Computes the critical domain size, the nucleation barrier, and
the number of receptors per domain, and compares to super-resolution
measurements (~70–80 nm, ~20 receptors). With a line tension of ~1 pN the model
gives a ~78 nm domain and a ~29 kBT barrier — implying nanodomain formation is
**activity-gated** (must be nucleated), consistent with LTP being an induced
event.

Output: `membrane_nucleation_2d.pdf/.png`

---

## Running

```bash
pip install -r requirements.txt
python kramers_escape_sim.py
python membrane_nucleation_2d.py
```

`kramers_escape_sim.py` runs a stochastic simulation and takes a minute or two.
Random seeds are fixed for reproducibility; change them to resample.

## Requirements
- Python ≥ 3.9
- numpy ≥ 1.24 (uses `numpy.trapezoid`; on older NumPy replace with `numpy.trapz`)
- matplotlib ≥ 3.6

## Parameters and units
All energies are in units of k_B T (k_B T = 4.28 × 10⁻²¹ J at 310 K). Parameter
choices (line tension, binding energies, areal densities) are stated at the top
of each script with the ranges they are drawn from; several are estimates, not
measured values for the PSD specifically, which is why the paper states its
predictions as scalings. See the paper's parameter table and limitations section.

## Citation
If you use this code, please cite the accompanying paper. (A more speculative
companion line of work is referenced in the paper's appendix and is clearly
labelled as conjecture there.)

## License
Suggested: MIT (add a `LICENSE` file). Choose whatever suits your posting venue.
