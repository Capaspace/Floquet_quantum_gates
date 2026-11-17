# Legacy code (original submission)

`cms_gaussian_noise_GPU_penalty_coupling_v4.ipynb` is the notebook used for the first version of the paper
(v3.2/v4 of the optimiser). It is kept unchanged for reference; the exact state of the repository at that
time is available under the git tag `v1-paper`.

Main differences with respect to the current code (`floquet/`):

| | legacy (v4) | current |
|---|---|---|
| units | $H = -\omega/2\,\sigma_z + J\,\sigma_x\sigma_x$ **without** $2\pi$ ($J = 0.5$ ≈ 500 MHz) | explicit $2\pi$, GHz and ns |
| model | two-level qubits | three-level transmons ($\alpha = -0.2$ GHz), qubit model for comparison |
| fidelity | amplitude $|\langle tgt|\psi_n\rangle|$ | population $|\langle tgt|\psi_n\rangle|^2$ |
| parameters | unconstrained ($\sigma \approx 0.16$ ns, $A \approx 24.5$, couplers between the qubits) | physical constraints by construction |
| kick integration | 2nd-order Strang, float32 diagonalisation | 4th-order Yoshida with exact Gaussian integral, float64 diagonalisation |
| noise | Monte Carlo on $\omega_i$, $J_i$ | quasi-static noise + Lindblad decoherence, noise-aware objective |

The legacy notebook optimises the qubit frequencies, the kick amplitude $A$, the pulse width $\sigma$ and the
period $T$ with CMA-ES on GPU, with a gated Monte Carlo robustness term
$\text{Score} = -[(1-\beta)F_\text{nom} + \beta F_{p20}]$ evaluated only for candidates above a fidelity threshold.
