# Floquet quantum gates with Gaussian flux kicks

Numerical design of **Floquet (periodically kicked) quantum gates** on superconducting circuits with tunable
couplers. A periodic train of Gaussian flux pulses on the couplers generates, after a few Floquet periods,
an excitation transfer between qubits. The code optimises the circuit and pulse parameters under
**physical constraints** and checks the resulting gates against a realistic transmon model, decoherence
and quasi-static noise.

This is the code accompanying the revised version of the paper *"Implementation of quantum gates by Floquet
analysis of kicked quantum system"*. The code of the original submission is kept in [`legacy/`](legacy/)
and under the git tag [`v1-paper`](../../tree/v1-paper).

---

## Systems

| System | Sites | Gate | Module / notebook |
|---|---|---|---|
| 7-site chain | q0 – c1 – q2 – c3 – q4 – c5 – q6 | two-excitation transfers `1100→0011`, `1010→0101`, `1001→0110` | [`floquet/chain.py`](floquet/chain.py), [`notebooks/seven_site_chain.ipynb`](notebooks/seven_site_chain.ipynb) |
| 3-site system (Table 1 of the paper) | q1 – bus – q2 | single-excitation transfer `10→01` | [`floquet/three_qubit.py`](floquet/three_qubit.py), [`notebooks/three_qubit_system.ipynb`](notebooks/three_qubit_system.ipynb) |

## Model

Every site is a **three-level transmon** (weakly anharmonic oscillator, $\alpha = -0.2$ GHz):

$$
\frac{H(t)}{h} = \sum_i \Big[f_i\, n_i + \frac{\alpha}{2}\, n_i(n_i-1)\Big]
 + \sum_{\langle ij\rangle} g_{ij}\,(a_i+a_i^\dagger)(a_j+a_j^\dagger)
 - 2A \sum_k e^{-(t-t_k)^2/2\sigma^2} \sum_{c} n_c ,
$$

with frequencies in GHz and times in ns ($2\pi$ explicit). The flux pulse lowers the coupler frequency,
$f_c(t) = f_c - 2A\,e^{-t^2/2\sigma^2}$. One Floquet period is the pulse window ($\pm 3\sigma$) plus a free
evolution $T$, i.e. $T_\text{per} = T + 6\sigma$; the gate is $U_F^{\,n}$ and the fidelity is the population
$|\langle \text{target}|U_F^{\,n}|\text{source}\rangle|^2$. A two-level (qubit) version of the model is kept for comparison.

**Physical constraints** (true by construction in the parametrisation, see `PHYS` in `floquet/chain.py`):

| Quantity | Range |
|---|---|
| qubit frequencies | 4.0 – 6.5 GHz |
| couplers | 0.7 – 3 GHz **above** the neighbouring qubits (dispersive regime), ≤ 9 GHz |
| couplings $g$ | 30 – 150 MHz |
| pulse width $\sigma$ | 2 – 10 ns |
| coupler excursion $2A$ | ≤ 2 GHz; the coupler stays ≥ 0.3 GHz above the qubits even at the pulse peak (no crossings) |
| gate duration | ≤ 450 ns |
| anharmonicity | penalty near the $|11\rangle\leftrightarrow|20\rangle$ resonance |
| Floquet-essential transfer | penalty if the transfer also happens **without** the pulse ($A=0$) |

## Numerical methods

- **Kick propagator**: 4th-order Yoshida splitting with the exact integral of the Gaussian (erf) on every
  sub-step; free evolution from an exact float64 diagonalisation; number of steps doubled until $|\Delta F| < 10^{-4}$.
- **Truncation**: parity sector with $N_\text{tot} \le 4$ (190 states for the 7-site chain), checked against $N_\text{tot} \le 6$.
- **Optimisation**: CMA-ES in normalised coordinates with IPOP/local restarts, GPU batching (PyTorch),
  checkpoints every 30 s (every long run is interruptible and resumable).
- **Decoherence**: Lindblad equation on both parity sectors with exact local amplitude-damping and dephasing
  channels (default $T_1 = 50\,\mu$s, $T_\varphi = 80\,\mu$s for qubits; $T_1 = 10\,\mu$s, $T_\varphi = 20\,\mu$s for couplers).
- **Quasi-static noise**: relative model $f_i \to f_i(1+\eta Z)$, $g_i \to g_i(1+\eta Z)$ (η scan, 128 samples),
  and a noise-aware objective $\big[\langle F\rangle_\eta - 0.1\,\mathrm{Var}_\eta F\big]\,e^{-t_\text{gate}/\tau_\text{dec}}$.

## Results shipped with the repository

Transmon model ($\alpha = -0.2$ GHz). $C$ = Floquet contribution (fidelity with pulse minus the best fidelity
without pulse within 1.5 µs).

| Transition | File | F | n | gate | C | F (Lindblad) | ⟨F⟩ at η = 1e-4 |
|---|---|---|---|---|---|---|---|
| 1100 → 0011 | `results/step3opt/v5_step3opt_1100_0011_local.json` | 0.9956 | 4 | 253 ns | 0.98 | 0.982 | 0.80 |
| 1100 → 0011 (noise-optimised) | `results/step3opt/v5_step3opt_1100_0011.json` | 0.9845 | 3 | 130 ns | 0.95 | 0.978 | 0.93 |
| 1010 → 0101 | `results/step3opt/v5_step3opt_1010_0101.json` | 0.9889 | 7 | 237 ns | 0.93 | 0.976 | 0.97 |
| 1001 → 0110 | `results/step2b/v5_step2b_1001_0110.json` | 0.9950 | 3 | 66 ns | 0.95 | 0.991 | 0.99 |
| 10 → 01 (3 qubits) | `results/step4/v5_step4_10_01.json` | 0.9992 | 3 | 67 ns | 0.90 | 0.997 | 0.996 |

$\eta = 10^{-4}$ corresponds to a frequency error $\delta f_q \approx 0.4$–$0.65$ MHz. The corresponding
analyses (Lindblad, η scan) are in `analysis/`. For comparison, the original Table 1 (3 qubits, bus between
the qubits, $g = 0.4$, delta kick, 447 cycles) gives $F = 0.99998$ in the qubit model but $F = 0.978$ in the
transmon model.

## Installation

```bash
git clone https://github.com/Capaspace/Floquet_quantum_gates.git
cd Floquet_quantum_gates
pip install -r requirements.txt
jupyter lab notebooks/
```

A CUDA GPU is strongly recommended for the 7-site optimisations (the analysis cells and the 3-qubit system
run fine on a CPU). PyTorch with CUDA support must be installed following the instructions at
https://pytorch.org for your platform.

## Usage

Open a notebook from the `notebooks/` folder. Each step has a parameter cell and a run cell; the analysis and
plotting cells work directly on the shipped JSON files, so the figures can be reproduced without re-running
the optimisations. From Python:

```python
from floquet import chain as fv
import json
o = json.load(open(fv.path_for('v5_step2b_1001_0110.json')))
tr = fv.qutrit('1001_0110')
curve, _ = fv.fidelity_curve(tr, [o['omegas_opt']], o['couplings'], [o['A_opt']], [o['sigma_opt']],
                             [o['T_opt']], prec=fv.P64, unit=fv.UNIT_PHYS, n_max=30)
print(curve[0, o['n_opt'] - 1])      # 0.995
```

Long runs (e.g. `fv.run_step3_opt`, `fv.run_step3_global`) save checkpoints in `checkpoints/`; calling the same
function again resumes where it stopped.

## Repository layout

```
floquet/
  chain.py            7-site chain: model, kick propagator, optimiser, Steps 1-3, Lindblad, noise
  three_qubit.py      3-site system (Table 1): same physics and analyses
notebooks/
  seven_site_chain.ipynb
  three_qubit_system.ipynb
results/              final solutions (JSON)           results/step<X>/v5_step<X>_<transition>.json
analysis/             noise/decoherence analyses (JSON), robustness-scan caches
legacy/               code of the original submission
```

Figures (`figures/`), data files (`data/`) and checkpoints (`checkpoints/`) are created by the notebooks
and are not tracked.

## Citation

If you use this code, please cite the paper (reference to be added) and this repository
(see [`CITATION.cff`](CITATION.cff)).

## License

MIT, see [`LICENSE`](LICENSE).
