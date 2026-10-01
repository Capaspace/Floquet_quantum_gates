# 7-Qubit Floquet System Optimization (Gaussian Pulses) - v3.2 Gated

This repository contains the Python implementation (`optimize_7qubit_floquet_gaussian_robust_gpu_v3_gated.py`) for the global optimization via **CMA-ES** (CMA Evolution Strategy) of control and frequency parameters for a 7-qubit quantum system driven by Gaussian pulses (qubit frequencies $\omega_i$, kick amplitude $A$, pulse width $\sigma$, and Floquet period $T$).

The codebase features a **GPU-accelerated Monte Carlo robustness optimization** framework via PyTorch, integrated with an **absolute-threshold gating mechanism** designed to drastically reduce evaluation runtimes.

---

## 📋 Table of Contents
1. [Key Features](#-key-features)
2. [Physical Model & Noise Strategy](#-physical-model--noise-strategy)
3. [Performance Optimizations & Gating (v3.2)](#-performance-optimizations--gating-v32)
4. [Requirements](#-requirements)
5. [Parameter Configuration](#-parameter-configuration)
6. [Execution & Output](#-execution--output)

---

## 🚀 Key Features

- **Optimization Algorithm:** 10-dimensional parameter space optimization using CMA-ES (`cma` library).
- **GPU/CPU Support:** Parallelized batched execution on GPU via PyTorch (`torch.linalg.eigh` and matrix exponential propagation) with automatic CPU fallback.
- **Stroboscopic Propagation:** Computation of the Floquet operator $U_{\text{Floquet}} = U_{\text{free}} \cdot U_{\text{kick}}$ using Trotter-Suzuki decomposition for Gaussian pulse shapes.
- **High-Precision Final Evaluation:** Post-optimization verification of the top candidates using exact diagonalization on CPU.

---

## ⚛️ Physical Model & Noise Strategy

### System Hamiltonian ($H_0$)
The system consists of 7 nearest-neighbor coupled qubits (Ising XX interaction) under longitudinal magnetic fields along $Z$:
$$H_0 = -\sum_{i=0}^{6} \frac{\omega_i}{2} \sigma_z^{(i)} + \sum_{i=0}^{5} J_i \sigma_x^{(i)} \sigma_x^{(i+1)}$$

The drive operator (Gaussian kick) is applied via the interaction operator $V$:
$$V = \sigma_z^{(1)} + \sigma_z^{(3)} + \sigma_z^{(5)}$$

### Noise Model (Physical Hardware Sensitivity)
Monte Carlo noise perturbations are applied **exclusively to real hardware parameters** prone to thermal drift or fabrication variance:
- **Qubit Frequencies ($\omega_i$):** $\omega_i \to \omega_i + \eta \cdot \vert{}\omega_i\vert{} \cdot \mathcal{N}(0, 1)$
- **Coupling Strengths ($J_i$):** $J_i \to J_i + \eta \cdot \vert{}J_i\vert{} \cdot \mathcal{N}(0, 1)$

Control parameters ($A, T, \sigma$) are assumed to be synthesized by external arbitrary waveform generators (AWGs) and are treated as noise-free.

---

## ⚡ Performance Optimizations & Gating (v3.2)

### Two-Phase Gating Evaluation
To avoid the computational overhead of Monte Carlo sensitivity sweeps ($N_{\text{PERT}}$ replicas) on low-fidelity candidates, the optimization pipeline employs a two-tier evaluation strategy:

1. **Phase A (Nominal Evaluation):** Calculates the nominal fidelity $F_{\text{nom}}$ across the entire population (`popsize = 32`).
2. **Phase B (Gated Robustness Evaluation):** The Monte Carlo sampling is triggered **only for candidate solutions exceeding an absolute fidelity threshold** (`FIDELITY_GATE_THRESHOLD = 0.90`).
   - If no candidates pass the gate, Phase B is entirely bypassed for that generation (zero extra computational cost).
   - For candidates below the threshold, $F_{p20}$ defaults to $F_{\text{nom}}$, maintaining landscape continuity for CMA-ES.

### Objective Function (Score)
$$\text{Score} = -\left[ (1 - \beta) F_{\text{nom}} + \beta F_{p20} \right]$$
*where $F_{p20}$ represents the 20th percentile fidelity sampled across $N_{\text{PERT}}$ Monte Carlo perturbations.*

---

## 🛠 Requirements

Install the necessary Python dependencies before running the scripts:

```bash
pip install numpy scipy torch cma tqdm
