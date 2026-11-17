"""
floquet.chain — two-excitation transfer in a 7-site chain q-c-q-c-q-c-q
driven by periodic Gaussian flux kicks on the couplers (Floquet analysis).

Hamiltonian (units selected by `unit`):
    H(t) = unit * [ sum_i -f_i/2 sz_i + sum_i g_i sx_i sx_{i+1} + A e^{-t^2/2s^2} sum_c sz_c ]
    unit = 1      -> legacy (v4) convention: f, g, A in rad/ns, t in ns            [Step 1]
    unit = 2*pi   -> f, g, A in GHz, t in ns (physical convention)                 [Step 2+]
Three-level transmon model (QutritTransition): same H with a, a^+ truncated to 3 levels and
anharmonicity alpha; the kick is -2 A f(t) sum_c n_c.

One Floquet cycle = kick (window [-3s, 3s]) + free evolution for a time T: period T + 6s.
Fidelity = population |<tgt|psi_n>|^2 after n cycles.

File layout
  0. Output folders (`path_for`)
  1. Chain, transitions, models (qubit / transmon)
  2. Static Hamiltonian, kick (4th-order Yoshida + exact Gaussian integral), F(n) curves
  3. Optimiser: normalised space, CMA-ES with restarts and CHECKPOINTS (interruptible)
  4. Step 1   — legacy (v4) physics and units
  5. Step 2   — physical ranges (GHz, ns), physics audit, transmon check
  6. Step 2b  — transfer produced by the drive (static penalty, gate <= 450 ns)
  7. Step 3   — analysis: Floquet contribution, Lindblad, quasi-static noise, recalibration
  8. Step 3   — noise-aware OPTIMISATION (relative noise eta + decoherence in the objective)
"""
import itertools
import json
import math
import os
import pickle
import re
import time
from contextlib import contextmanager

import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════════════
# 0. Output folders
# ══════════════════════════════════════════════════════════════════════════════
#   results/step1|step2|step2b|step3opt/      solutions        v5_step<X>_<name>.json
#   analysis/step3|step3b|step3opt/           noise/decoherence analyses (v5_step3_, v5_step3b_, v5_step3opt_analysis_)
#   analysis/robustness/                      eta scans for the robustness curves (v5_eta_*)
#   figures/step1|step2|step2b|step3opt/      PDFs of a solution (fidelity, populations, resolvent, FFT)
#   figures/noise/                            noise and decoherence PDFs (robustness curve, bands, F vs T1)
#   data/step<X>/                             time evolutions (.txt) and quasi-energies (.dat)
#   checkpoints/                              state of interrupted optimisations
# All paths are relative to DATA_ROOT (the repository root by default).
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STEP_RE = re.compile(r'v5_step(3opt_analysis|3opt|3b|3|2b|2|1)_')


def path_for(fn, base=None):
    """
    Full path (folder created if missing) for a file named `fn`; the folder is chosen from the name.
    If `fn` already contains a folder it is returned unchanged. ALWAYS read/write through this function,
    e.g. open(fv.path_for('v5_step2b_1100_0011.json')), plt.savefig(fv.path_for(name + '.pdf')).
    """
    if os.path.dirname(fn):
        return fn
    base = DATA_ROOT if base is None else base
    m = _STEP_RE.match(fn)
    tag = m.group(1) if m else None
    ext = os.path.splitext(fn)[1].lower()
    noisy = tag in ('3', '3b', '3opt_analysis') or any(k in fn for k in ('robustness', 'band', 'F_vs_T1'))
    if ext == '.json':
        if fn.startswith('v5_eta_'):
            d = 'analysis/robustness'
        elif tag in ('3', '3b'):
            d = f'analysis/step{tag}'
        elif tag == '3opt_analysis':
            d = 'analysis/step3opt'
        else:
            d = f'results/step{tag}' if tag else '.'
    elif ext in ('.pdf', '.png', '.svg'):
        d = 'figures/noise' if noisy else (f'figures/step{tag}' if tag else 'figures')
    elif ext in ('.txt', '.dat', '.csv', '.npy', '.npz'):
        d = f"data/step{tag.replace('_analysis', '')}" if tag else 'data'
    else:
        d = '.'
    d = os.path.normpath(os.path.join(base, d))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fn)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Chain, transitions, models
# ══════════════════════════════════════════════════════════════════════════════
GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if GPU else 'cpu')

N_SITES = 7
QUBITS = [0, 2, 4, 6]
COUPLERS = [1, 3, 5]
DIM = 2 ** N_SITES


def _bit(idx, site):
    """Bit of site `site` in a basis index (site 0 = most significant bit, as in v4)."""
    return (idx >> (N_SITES - 1 - site)) & 1


def bits_to_index(bits):
    return sum((1 << (N_SITES - 1 - i)) * int(b) for i, b in enumerate(bits))


def sites_to_index(sites):
    bits = [0] * N_SITES
    for s in sites:
        bits[s] = 1
    return bits_to_index(bits)


# diagonal sigma_z: |0> -> +1, |1> -> -1 (as in v4)
SZ_FULL = np.array([[1.0 - 2.0 * _bit(k, i) for k in range(DIM)] for i in range(N_SITES)])  # (7,128)
XX_FULL = np.zeros((N_SITES - 1, DIM, DIM))
for _i in range(N_SITES - 1):
    _mask = (1 << (N_SITES - 1 - _i)) | (1 << (N_SITES - 2 - _i))
    for _k in range(DIM):
        XX_FULL[_i, _k ^ _mask, _k] = 1.0
V_FULL = SZ_FULL[COUPLERS].sum(axis=0)  # (128,)

TRANSITIONS = {
    '1100_0011': ([0, 2], [4, 6]),
    '1010_0101': ([0, 4], [2, 6]),
    '1001_0110': ([0, 6], [2, 4]),
}


class Transition:
    """Qubit model: source/target and operators restricted to the parity sector (or the full space)."""

    def __init__(self, name, src_sites=None, tgt_sites=None, reduce_parity=True):
        if src_sites is None:
            src_sites, tgt_sites = TRANSITIONS[name]
        self.name = name
        self.src_full = sites_to_index(src_sites)
        self.tgt_full = sites_to_index(tgt_sites)
        if reduce_parity:
            par = bin(self.src_full).count('1') % 2
            sector = np.array([k for k in range(DIM) if bin(k).count('1') % 2 == par])
        else:
            sector = np.arange(DIM)
        pos = {int(k): j for j, k in enumerate(sector)}
        assert self.tgt_full in pos, "source and target lie in different parity sectors"
        self.sector = sector
        self.D = len(sector)
        self.src_i = pos[self.src_full]
        self.tgt_i = pos[self.tgt_full]
        self.sz = SZ_FULL[:, sector]                                       # (7, D)
        self.xx_flat = XX_FULL[:, sector][:, :, sector].reshape(N_SITES - 1, -1)  # (6, D*D)
        self.v = V_FULL[sector]                                            # (D,) kick operator
        self.leak_mask = np.zeros(self.D, dtype=bool)                      # no |2> levels
        self._torch_cache = {}

    def diag(self, f):
        """Diagonal part of H0 (without unit): -f_i/2 sz_i.  f: (B,7) -> (B,D)"""
        return -(f @ self.sz) / 2.0

    def torch_consts(self):
        """(v, |src>) on DEVICE in float64/complex128."""
        if not self._torch_cache:
            v = torch.tensor(self.v, dtype=torch.float64, device=DEVICE)
            src = torch.zeros(self.D, dtype=torch.complex128, device=DEVICE)
            src[self.src_i] = 1.0
            self._torch_cache = (v, src)
        return self._torch_cache


class QutritTransition(Transition):
    """
    Three-level transmon model:
        H0 = sum_i [ f_i n_i + alpha_i/2 n_i(n_i-1) ] + sum_i g_i (a_i+a_i^+)(a_{i+1}+a_{i+1}^+)
        kick: A f(t) * (-2 sum_c n_c)       (= A f(t) sum_c sz_c up to a constant)
    Restricted to levels 0,1 it coincides with the qubit model (-f/2 sz = f n - f/2; sx sx = X X).
    Basis: N_tot <= nmax with the N_tot parity of the source (the coupling changes N by 0 or 2);
    both_parities=True keeps all states with N_tot <= nmax (needed by Lindblad: T1 changes N).
    """

    def __init__(self, name, alpha_q=-0.2, alpha_c=-0.2, nmax=4, both_parities=False):
        src_sites, tgt_sites = TRANSITIONS[name]
        self.name = name
        par = len(src_sites) % 2
        states = [s for s in itertools.product(range(3), repeat=N_SITES)
                  if sum(s) <= nmax and (both_parities or sum(s) % 2 == par)]
        pos = {s: j for j, s in enumerate(states)}
        D = len(states)
        S = np.array(states)                                   # (D,7)
        self.states, self.D = states, D
        src = tuple(1 if k in src_sites else 0 for k in range(N_SITES))
        tgt = tuple(1 if k in tgt_sites else 0 for k in range(N_SITES))
        self.src_i, self.tgt_i = pos[src], pos[tgt]
        self.src_full, self.tgt_full = sites_to_index(src_sites), sites_to_index(tgt_sites)
        self.n = S.T.astype(np.float64)                        # (7,D)
        alpha = np.array([alpha_c if k in COUPLERS else alpha_q for k in range(N_SITES)])
        self.alpha = alpha
        self.anh = (alpha[:, None] / 2.0 * self.n * (self.n - 1)).sum(axis=0)   # (D,)
        self.v = -2.0 * self.n[COUPLERS].sum(axis=0)
        self.leak_mask = (S == 2).any(axis=1)
        XX = np.zeros((N_SITES - 1, D, D))                     # (a_i + a_i^+)(a_{i+1} + a_{i+1}^+)
        for j, s in enumerate(states):
            for i in range(N_SITES - 1):
                for di in (-1, 1):
                    ki = s[i] + di
                    if not 0 <= ki <= 2:
                        continue
                    ai = math.sqrt(max(s[i], ki))
                    for dj in (-1, 1):
                        kj = s[i + 1] + dj
                        if not 0 <= kj <= 2:
                            continue
                        t = list(s); t[i] = ki; t[i + 1] = kj; t = tuple(t)
                        if t in pos:
                            XX[i, pos[t], j] = ai * math.sqrt(max(s[i + 1], kj))
        self.xx_flat = XX.reshape(N_SITES - 1, -1)
        self._torch_cache = {}

    def diag(self, f):
        return f @ self.n + self.anh[None, :]


def qutrit(name, nmax=4, both_parities=False):
    """Transmon model with the anharmonicity in PHYS (qubits and couplers)."""
    return QutritTransition(name, PHYS['ALPHA'], PHYS['ALPHA'], nmax, both_parities)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Static Hamiltonian, kick, F(n) curves
# ══════════════════════════════════════════════════════════════════════════════
class Prec:
    def __init__(self, f32):
        self.f = torch.float32 if f32 else torch.float64
        self.c = torch.complex64 if f32 else torch.complex128


P32, P64 = Prec(True), Prec(False)


def _cis(x):
    return torch.polar(torch.ones_like(x), x)


def _t64(a):
    return torch.as_tensor(np.ascontiguousarray(np.atleast_1d(a)), dtype=torch.float64, device=DEVICE)


def build_H0(tr, f, g, unit):
    """f: (B,7), g: (B,6) or (6,) -> (B,D,D) float64 (real symmetric), already multiplied by unit."""
    f = np.atleast_2d(np.asarray(f, dtype=np.float64))
    g = np.asarray(g, dtype=np.float64)
    B = f.shape[0]
    if g.ndim == 1:
        g = np.broadcast_to(g, (B, N_SITES - 1))
    H = (g @ tr.xx_flat).reshape(B, tr.D, tr.D)
    idx = np.arange(tr.D)
    H[:, idx, idx] += tr.diag(f)
    return unit * H


EIGH_ON_GPU = False


def diagonalize(H):
    """Always in float64: the transfer exploits near-degeneracies, where a float32 eigh gives
    wrong eigenvectors and CMA ends up optimising the numerical error."""
    Hc = np.asarray(H, dtype=np.float64)
    if EIGH_ON_GPU:
        return torch.linalg.eigh(torch.from_numpy(Hc).to(DEVICE))
    ev, evc = np.linalg.eigh(Hc)
    return (torch.from_numpy(ev).to(DEVICE),
            torch.from_numpy(np.ascontiguousarray(evc)).to(DEVICE))


def pick_eigh_backend(D=64, B=64):
    """Micro-benchmark GPU vs CPU for a batched eigh (as in v4)."""
    global EIGH_ON_GPU
    if not GPU:
        EIGH_ON_GPU = False
        return
    rng = np.random.default_rng(0)
    M = rng.normal(size=(B, D, D)).astype(np.float32)
    H = (M + M.transpose(0, 2, 1)) / 2
    Ht = torch.from_numpy(H).to(DEVICE)
    torch.linalg.eigh(Ht); torch.cuda.synchronize()
    t = time.perf_counter(); torch.linalg.eigh(Ht); torch.cuda.synchronize()
    tg = time.perf_counter() - t
    t = time.perf_counter(); np.linalg.eigh(H)
    tc = time.perf_counter() - t
    EIGH_ON_GPU = tg < tc
    print(f"[eigh] {B}x{D}x{D}: GPU {tg*1e3:.1f} ms, CPU {tc*1e3:.1f} ms -> {'GPU' if EIGH_ON_GPU else 'CPU'}")


# ── Gaussian kick: Yoshida-4 on top of Strang, with the exact integral of the Gaussian ──────
_Y1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_Y0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

# Number of steps: calibrated with a convergence test (f64, 4 regimes of A, sigma). With this rule
# and rounding to powers of 2 the error on F_pop is < 1e-4 (typically ~1e-6).
TROTTER_THETA = 1.0        # rad: maximum kick phase per step (unit*A*n_c*h)
TROTTER_HMAX = 0.2         # maximum step for unit = 1 (Step 1)
TROTTER_HMAX_PHYS = 0.02   # ns, maximum step for unit = 2pi: the error comes from the counter-rotating
                           # terms of sx sx (f_c + f_q ~ 14 GHz) -> h * 2pi * f_max ~ 1
TROTTER_MIN = 16
TROTTER_MAX = 4096
WINDOW = 3.0               # the kick is integrated over [-WINDOW*s, +WINDOW*s] (6 sigma, as in v4)
KICK_CHUNK_BYTES = 1_500_000_000   # maximum GPU memory per block of steps inside the kick


def n_trotter_required(A, sigma, unit, theta=None, hmax=None):
    theta = TROTTER_THETA if theta is None else theta
    hcap = (TROTTER_HMAX if hmax is None else hmax) if unit == 1.0 else TROTTER_HMAX_PHYS
    A = np.abs(np.asarray(A, dtype=np.float64))
    sigma = np.asarray(sigma, dtype=np.float64)
    h = np.minimum(hcap, theta / (unit * A * len(COUPLERS) + 1e-12))
    return np.ceil(2 * WINDOW * sigma / h).astype(int)


def _levels(req):
    """Round up to powers of 2 (to group candidates)."""
    req = np.clip(req, TROTTER_MIN, TROTTER_MAX)
    return np.minimum(TROTTER_MAX, 2 ** np.ceil(np.log2(req))).astype(int)


def n_trotter(A, sigma, unit, factor=1):
    """Number of steps for a single candidate (factor = 2 for the float64 checks)."""
    return int(_levels(n_trotter_required([A], [sigma], unit))[0]) * factor


def kick_unitary(tr, ev, evc, A, sigma, n_tr, prec, unit):
    """
    U_kick for a batch: ev (B,D), evc (B,D,D) real (H0 already scaled by unit).
    A, sigma: (B,) float64 tensors. Returns (B,D,D) complex.
    Strang-Yoshida, step matrix  S_j = K_c P3_j K_b P2_j K_b P1_j  (last step: K_a instead of K_c),
    U = S_{n-1} ... S_0 K_a. The S_j of a block are built together and multiplied as a TREE:
    the round-off error grows like log(n) instead of n (decisive in complex64).
    """
    v, _ = tr.torch_consts()
    evc_c = evc.to(torch.complex128)
    evc_d = evc_c.transpose(-2, -1).conj().contiguous()

    sig = sigma.double()
    h = (2 * WINDOW * sig / n_tr)                                    # (B,)
    t0 = -WINDOW * sig
    k = torch.arange(n_tr, dtype=torch.float64, device=DEVICE)
    offs = torch.tensor([0.0, _Y1, _Y1 + _Y0, 1.0], dtype=torch.float64, device=DEVICE)  # Yoshida sub-steps
    tb = t0[None, None, :] + (k[:, None, None] + offs[None, :, None]) * h[None, None, :]   # (n,4,B)
    pref = (A.double() * sig * math.sqrt(math.pi / 2.0))[None, None, :]
    Phi = pref * torch.special.erf(tb / (math.sqrt(2.0) * sig[None, None, :]))
    dPhi = (Phi[:, 1:, :] - Phi[:, :-1, :]) * unit                  # (n,3,B) exact integral per sub-step

    evf = ev.double()
    hf = h[:, None]
    K = lambda tau: torch.bmm(evc_c * _cis(-evf * tau).unsqueeze(1), evc_d)   # evc e^{-iE tau} evc^T
    K_a = K(_Y1 * hf / 2)              # initial / final edge
    K_b = K((_Y1 + _Y0) * hf / 2)      # between the inner sub-steps (tau < 0)
    K_c = K(_Y1 * hf)                  # between consecutive steps
    P = _cis(-dPhi[..., None] * v[None, None, None, :])             # kick phases (n,3,B,D) complex128
    Kb, Kc, Ka = K_b.to(prec.c), K_c.to(prec.c), K_a.to(prec.c)

    B, D = ev.shape
    C = int(max(1, min(n_tr, KICK_CHUNK_BYTES // (4 * B * D * D * 16))))
    C = 1 << (C.bit_length() - 1)                                    # power of 2
    U = Ka
    for j0 in range(0, n_tr, C):
        j1 = min(n_tr, j0 + C)
        m = j1 - j0
        X = (P[j0:j1, 1].unsqueeze(-1) * K_b.unsqueeze(0) * P[j0:j1, 0].unsqueeze(-2)).to(prec.c)
        X = torch.matmul(Kb, X)
        X = P[j0:j1, 2].unsqueeze(-1).to(prec.c) * X
        if j1 == n_tr:                                                # last step: K_a
            Y = torch.matmul(Kc, X[:-1]) if m > 1 else X[:0]
            X = torch.cat([Y, torch.matmul(Ka, X[-1:])], 0)
        else:
            X = torch.matmul(Kc, X)
        while X.shape[0] > 1:                                         # tree product X[m-1]...X[0]
            last = None
            if X.shape[0] % 2:
                last, X = X[-1:], X[:-1]
            X = torch.matmul(X[1::2], X[0::2])
            if last is not None:
                X = torch.cat([X, last], 0)
        U = torch.bmm(X[0], U)
    if prec.c == torch.complex64:
        # In complex64 U accumulates a NON-unitary error (~4e-4 after 1000 steps) that F^n amplifies.
        # Projection onto the closest unitary: 2 Newton-Schulz iterations in complex128.
        U = U.to(torch.complex128)
        I3 = 3.0 * torch.eye(U.shape[-1], dtype=U.dtype, device=U.device)
        for _ in range(2):
            U = 0.5 * torch.bmm(U, I3 - torch.bmm(U.mH, U))
    return U


def floquet_scan(tr, ev, evc, U_kick, T, n_max):
    """F_pop(n) = |<tgt|(Ufree Ukick)^n|src>|^2 for n=1..n_max, all in complex128. (B, n_max)"""
    _, src = tr.torch_consts()
    evc_c = evc.to(torch.complex128)
    U_free = torch.bmm(evc_c * _cis(-ev.double() * T.double()[:, None]).unsqueeze(1),
                       evc_c.transpose(-2, -1).conj())
    F = torch.bmm(U_free, U_kick.to(torch.complex128))
    B = ev.shape[0]
    psi = src.unsqueeze(0).expand(B, -1).unsqueeze(2).clone()
    out = torch.empty(B, n_max, dtype=torch.float64, device=DEVICE)
    for n in range(n_max):
        psi = torch.bmm(F, psi)
        out[:, n] = psi[:, tr.tgt_i, 0].abs() ** 2
    return out


def fidelity_curve(tr, f, g, A, sigma, T, prec=P32, unit=1.0, n_max=30, n_tr=None, theta=None, hmax=None):
    """
    F_pop(n) curves, n=1..n_max, for a batch of candidates.
    f (B,7), g (B,6) or (6,), A/sigma/T (B,) or scalars. Returns (np.ndarray (B, n_max), n_tr used).
    """
    f = np.atleast_2d(np.asarray(f, dtype=np.float64))
    B = f.shape[0]
    A, sigma, T = (np.broadcast_to(np.asarray(x, dtype=np.float64), (B,)) for x in (A, sigma, T))
    g = np.asarray(g, dtype=np.float64)
    g = np.broadcast_to(g, (B, N_SITES - 1)) if g.ndim == 1 else g
    lev = _levels(n_trotter_required(A, sigma, unit, theta, hmax)) if n_tr is None else np.full(B, int(n_tr))
    ev, evc = diagonalize(build_H0(tr, f, g, unit))
    out = np.empty((B, n_max))
    for L in np.unique(lev):
        idx = np.where(lev == L)[0]
        it = torch.as_tensor(idx, device=DEVICE)
        Uk = kick_unitary(tr, ev[it], evc[it], _t64(A[idx]), _t64(sigma[idx]), int(L), prec, unit)
        out[idx] = floquet_scan(tr, ev[it], evc[it], Uk, _t64(T[idx]), n_max).cpu().numpy()
    return out, lev


def fidelity_curve_batched(tr, f, g, A, sigma, T, batch=64, **kw):
    """fidelity_curve in blocks of `batch` candidates (limited GPU memory)."""
    f = np.atleast_2d(f)
    B = len(f)
    bc = lambda x, shape: np.broadcast_to(np.asarray(x, dtype=np.float64), shape)
    A, sigma, T = bc(A, (B,)), bc(sigma, (B,)), bc(T, (B,))
    g = np.asarray(g, dtype=np.float64)
    g = np.broadcast_to(g, (B, N_SITES - 1)) if g.ndim == 1 else g
    parts = [fidelity_curve(tr, f[j:j + batch], g[j:j + batch], A[j:j + batch], sigma[j:j + batch],
                            T[j:j + batch], **kw)[0] for j in range(0, B, batch)]
    return np.concatenate(parts, axis=0)


def best_over_n(curves, n_min, n_max_each=None, tol=0.0):
    """
    Choose n for each curve between n_min and n_max_each (inclusive; None = end of the curve).
    tol = 0: n of the maximum; tol > 0: the FIRST n with F >= F_max - tol (shorter gate for the same F).
    Returns (F(n), n) with 1-based n; F = 0 if no n is allowed.
    """
    c = np.array(curves, dtype=float, copy=True)
    n = np.arange(1, c.shape[1] + 1)
    hi = c.shape[1] if n_max_each is None else np.atleast_1d(n_max_each)
    c[(n[None, :] < n_min) | (n[None, :] > np.reshape(hi, (-1, 1)))] = -1.0
    if tol > 0:
        j = np.argmax(c >= (c.max(axis=1) - tol)[:, None], axis=1)
    else:
        j = np.argmax(c, axis=1)
    return np.maximum(c[np.arange(len(c)), j], 0.0), j + 1


def floquet_operator(tr, f, g, A, sigma, T, unit, n_tr=None, prec=P64):
    """Floquet operator U_free(T) U_kick for a single candidate (numpy, complex128)."""
    f = np.atleast_2d(np.asarray(f, dtype=np.float64))
    n_tr = n_trotter(A, sigma, unit, 2) if n_tr is None else n_tr
    ev, evc = diagonalize(build_H0(tr, f, np.asarray(g, dtype=np.float64), unit))
    Uk = kick_unitary(tr, ev, evc, _t64(A), _t64(sigma), n_tr, prec, unit).to(torch.complex128)
    evc_c = evc.to(torch.complex128)
    Uf = torch.bmm(evc_c * _cis(-ev.double() * float(T)).unsqueeze(1), evc_c.transpose(-2, -1).conj())
    return torch.bmm(Uf, Uk)[0].cpu().numpy()


def floquet_operator_full(name, f, g, A, sigma, T, unit=1.0, n_tr=None):
    """Floquet operator in the qubit model, full 128x128 space (for plots/analysis)."""
    tr = Transition(name, reduce_parity=False)
    return floquet_operator(tr, f, g, A, sigma, T, unit, n_tr=n_tr), tr


# ══════════════════════════════════════════════════════════════════════════════
# 3. Optimiser: normalised space [0,1], CMA-ES with restarts and checkpoints
# ══════════════════════════════════════════════════════════════════════════════
class Space:
    def __init__(self, entries):
        """entries: list of (name, low, high, log)"""
        self.entries = entries
        self.names = [e[0] for e in entries]
        self.dim = len(entries)

    def decode(self, X):
        X = np.clip(np.atleast_2d(np.asarray(X, dtype=np.float64)), 0.0, 1.0)
        out = {}
        for j, (nm, lo, hi, lg) in enumerate(self.entries):
            if lg:
                out[nm] = np.exp(np.log(lo) + X[:, j] * (np.log(hi) - np.log(lo)))
            else:
                out[nm] = lo + X[:, j] * (hi - lo)
        return out

    def encode(self, params):
        x = np.empty(self.dim)
        for j, (nm, lo, hi, lg) in enumerate(self.entries):
            v = params[nm]
            x[j] = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (v - lo) / (hi - lo)
        return x

    def to_json(self):
        return [{'name': n, 'low': lo, 'high': hi, 'log': lg} for n, lo, hi, lg in self.entries]


CKPT_DIR = 'checkpoints'     # relative to out_dir (DATA_ROOT by default)
CKPT_EVERY_S = 30.0          # save at most every 30 s (and at the end of every restart)


def _ckpt_save(path, state):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(state, fh)
    os.replace(tmp, path)          # atomic write: an interruption never corrupts the checkpoint


def _ckpt_load(path, signature, log):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as fh:
            st = pickle.load(fh)
    except Exception as e:                         # corrupted file: start again
        log(f"    checkpoint {path} unreadable ({e}): starting from scratch")
        return None
    old = st.get('signature') or {}
    if old != signature:
        diff = sorted(k for k in set(old) | set(signature) if old.get(k) != signature.get(k))
        log(f"    checkpoint {os.path.basename(path)} has different parameters ({', '.join(diff)}): "
            f"ignored, starting from scratch")
        return None
    return st


def run_cma(objective, dim, warm_starts=(), popsize=32, sigma0=0.25, sigma0_warm=0.08,
            max_evals=60000, max_time=None, ftarget=None, max_restarts=40,
            stall_gens=80, stall_tol=1e-5, seed=1, log=print, local_restarts=True, checkpoint=None,
            sig_extra=None, random_restarts=True):
    """
    Minimise objective(X) -> (fit (lam,), extra dict) with X in [0,1]^dim.
    Restarts: first the warm starts (sigma0_warm), then alternating random (IPOP: population x2) and
    local around the best point. Returns {'x', 'f', 'extra', 'evals', 'time', 'hist'}.

    checkpoint = .pkl path: the full state (CMA, restart, best, counters, RNG) is saved every
    CKPT_EVERY_S seconds. If the file exists (same configuration) the optimisation RESUMES from it;
    if the optimisation had already finished the result is returned immediately. max_time counts the
    total time over all sessions. sig_extra: objective parameters (e.g. noise level) that must match
    to resume; if they changed the checkpoint is ignored and the run starts from scratch.
    random_restarts=False: after the warm starts only LOCAL restarts around the best point, with sigma0_warm
    alternating with sigma0_warm/3 (useful when a random point is hopeless, e.g. narrow resonances).
    """
    import cma
    sig = dict(dim=dim, popsize=popsize, seed=seed, n_warm=len(warm_starts), sigma0=sigma0,
               sigma0_warm=sigma0_warm, **(sig_extra or {}))
    if not random_restarts:
        sig['random_restarts'] = False
    st = _ckpt_load(checkpoint, sig, log)
    if st and st.get('done'):
        log(f"    [checkpoint] optimisation already completed: f={st['best']['f']:.6f}")
        return st['best']
    rng = np.random.default_rng(seed)
    best = {'x': None, 'f': np.inf, 'extra': None}
    evals, elapsed0, hist, lam, r0, es_resume = 0, 0.0, [], popsize, 0, None
    if st:
        rng.bit_generator.state = st['rng']
        best, evals, elapsed0, hist, lam, r0 = st['best'], st['evals'], st['elapsed'], st['hist'], st['lam'], st['r']
        es_resume = st.get('es')
        np.random.set_state(st['np_global'])          # CMA also uses numpy's global generator
        log(f"    [checkpoint] resuming: restart {r0}, {evals} evaluations, {elapsed0:.0f} s, best f={best['f']:.6f}")
    t_start = time.perf_counter() - elapsed0
    elapsed = lambda: time.perf_counter() - t_start
    last_save = [time.perf_counter()]
    warm = [np.clip(np.asarray(w, float), 0, 1) for w in warm_starts]

    def save(r, es, extra_state=None, force=False, done=False):
        if not checkpoint or (not force and time.perf_counter() - last_save[0] < CKPT_EVERY_S):
            return
        state = dict(signature=sig, rng=rng.bit_generator.state, np_global=np.random.get_state(), best=best,
                     evals=evals, elapsed=elapsed(), hist=hist, lam=lam, r=r, es=es, done=done)
        if extra_state:
            state.update(extra_state)
        _ckpt_save(checkpoint, state)
        last_save[0] = time.perf_counter()

    def budget_done():
        return (ftarget is not None and best['f'] <= ftarget) or evals >= max_evals or \
               (max_time is not None and elapsed() > max_time)

    for r in range(r0, max_restarts):
        if budget_done():
            break
        if es_resume is not None:
            es, kind = es_resume, 'resumed'
            r_best, last_imp, gen = st['r_best'], st['last_imp'], st['gen']
            es_resume = None
            log(f"--- restart {r}: resumed from checkpoint (gen {gen})")
        else:
            if r < len(warm):
                x0, s0, lam_r, kind = warm[r], sigma0_warm, popsize, 'warm'
            elif not random_restarts and best['x'] is not None:
                s0 = sigma0_warm if (r - len(warm)) % 2 == 0 else sigma0_warm / 3
                x0, lam_r, kind = best['x'], popsize, 'local'
            elif local_restarts and best['x'] is not None and (r - len(warm)) % 2 == 1:
                x0, s0, lam_r, kind = best['x'], sigma0_warm, popsize, 'local'
            else:
                x0, s0, lam_r, kind = rng.uniform(0.1, 0.9, dim), sigma0, lam, 'random'
                lam = min(lam * 2, 256)
            es = cma.CMAEvolutionStrategy(list(x0), s0, {
                'bounds': [0.0, 1.0], 'popsize': lam_r, 'verbose': -9,
                'seed': int(rng.integers(1, 2**31 - 1)), 'tolfun': 1e-7, 'tolx': 1e-7,
                'tolstagnation': 10**7})
            log(f"--- restart {r}: {kind} x0, popsize={lam_r}, sigma0={s0}")
            if kind != 'random':
                # CMA never evaluates x0: evaluate it and inject it (the peaks can be extremely narrow)
                fit0, extra0 = objective(np.asarray(x0)[None])
                evals += 1
                if fit0[0] < best['f']:
                    best = {'x': np.asarray(x0).copy(), 'f': float(fit0[0]),
                            'extra': {k: (v[0] if hasattr(v, '__len__') else v) for k, v in extra0.items()}}
                    log(f"    x0: f={best['f']:.6f}")
                es.inject([list(x0)], force=True)
            r_best, last_imp, gen = np.inf, 0, 0
        while not es.stop():
            X = np.array(es.ask())
            fit, extra = objective(X)
            es.tell(list(X), list(fit))
            evals += len(X); gen += 1
            i = int(np.argmin(fit))
            if fit[i] < r_best - stall_tol:
                r_best, last_imp = fit[i], gen
            if fit[i] < best['f']:
                best = {'x': X[i].copy(), 'f': float(fit[i]),
                        'extra': {k: (v[i] if hasattr(v, '__len__') else v) for k, v in extra.items()}}
                log(f"[{evals:>6} | {elapsed():7.1f}s] r{r} gen {gen:4d}  best f={best['f']:.6f}  "
                    + "  ".join(f"{k}={best['extra'][k]:.4g}" for k in best['extra']))
            hist.append((evals, float(fit[i]), best['f']))
            save(r, es, dict(r_best=r_best, last_imp=last_imp, gen=gen))
            if budget_done() or gen - last_imp > stall_gens:
                break
        log(f"    end of restart {r}: gen={gen}, restart best f={r_best:.6f}, stop={dict(es.stop()) or 'stall/target'}")
        if budget_done():
            break
        save(r + 1, None, force=True)                     # restart finished: resume from the next one
    best = dict(best, evals=evals, time=elapsed(), hist=hist)
    save(max_restarts, None, force=True, done=True)
    return best


def ckpt_path(out_dir, tag, name, phase):
    d = os.path.join(DATA_ROOT if out_dir is None else out_dir, CKPT_DIR)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{tag}_{name}_{phase}.pkl')


def clear_checkpoints(out_dir, tag, name):
    d = os.path.join(DATA_ROOT if out_dir is None else out_dir, CKPT_DIR)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.startswith(f'{tag}_{name}_'):
                os.remove(os.path.join(d, fn))


# ══════════════════════════════════════════════════════════════════════════════
# float64 validation (qubit model, full 128-dim space, Trotter convergence n vs 2n)
# ══════════════════════════════════════════════════════════════════════════════
N_MIN_CYCLES = 3      # n >= 3 (as in v4: step >= 2 with 0-based steps)
N_MAX_CYCLES = 30


def validate(name, f, g, A, sigma, T, unit=1.0, n_min=N_MIN_CYCLES, n_max=N_MAX_CYCLES):
    tr = Transition(name, reduce_parity=False)
    L = n_trotter(A, sigma, unit)

    def run(n_tr):
        c, _ = fidelity_curve(tr, np.atleast_2d(f), g, [A], [sigma], [T], prec=P64,
                              unit=unit, n_max=n_max, n_tr=n_tr)
        Fb, nb = best_over_n(c, n_min)
        return float(Fb[0]), int(nb[0]), c[0]

    prev = run(L)
    while True:                       # double until |F(n_tr) - F(2 n_tr)| < 1e-4
        cur = run(2 * L)
        err = abs(prev[0] - cur[0])
        if err < 1e-4 or 2 * L >= 16384:
            break
        prev, L = cur, 2 * L
    F2, n2, curve = cur
    return {'F_pop': F2, 'n_opt': n2, 'F_amp': math.sqrt(F2), 'n_trotter': 2 * L,
            'trotter_err': err, 'curve': curve.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# 4. STEP 1 — legacy (v4) physics and units, fixed couplings, new optimiser
# ══════════════════════════════════════════════════════════════════════════════
COUPLINGS_V4 = np.array([0.497, 0.499, 0.501, 0.502, 0.5, 0.498])

SPACE_STEP1 = Space(
    [(f'f{i}', 4.0, 7.0, False) if i in QUBITS else (f'f{i}', 7.0, 9.0, False) for i in range(N_SITES)]
    + [('A', 0.02, 40.0, True), ('sigma', 0.1, 10.0, False), ('T', 40.0, 150.0, False)])

# Legacy (v4 and earlier) JSON solutions used as optional warm starts for Step 1 (not shipped)
OLD_RESULTS_DIR = os.path.join(DATA_ROOT, 'legacy', 'old_results')
WARM_JSON = {
    '1100_0011': ['1100_0011_gaussiano_sigma.json', '1100_0011_gaussiano_3.json',
                  'opt_results_gaussian_robust_gpu_v4.json'],
    '1010_0101': ['1010_0101_gaussiano_sigma.json', '1010_0101_gaussiano.json'],
    '1001_0110': ['1001_0110_gaussiano_sigma.json', '1001_0110_gaussiano.json'],
}
TIE_DETUNING = 0.2   # for 'parallel swaps': f_dep = f_src + d, d in [-0.2, 0.2]


def resonance_ties(name):
    """
    Resonance constraints for phase A, as {dependent_site: source_site} (f_dep = f_src).
      * if the mirror i -> 6-i maps the source onto the target: f_i = f_{6-i}
      * otherwise each source qubit is put on resonance with the adjacent target qubit
        (e.g. 1001->0110 = two parallel swaps q0->q2 and q6->q4: f2 = f0, f4 = f6)
    """
    src, tgt = TRANSITIONS[name]
    if sorted(N_SITES - 1 - s for s in src) == sorted(tgt):
        return {k: N_SITES - 1 - k for k in range(N_SITES // 2 + 1, N_SITES)}, 'mirror'
    ties = {}
    for s in src:
        for t in tgt:
            if abs(s - t) == 2 and t not in ties:
                ties[t] = s
    return ties, 'parallel swaps'


def tied_space(ties, offsets=False):
    """Reduced space. With offsets=True the tied sites become a small detuning d_k."""
    ent = [e for e in SPACE_STEP1.entries if not (e[0].startswith('f') and int(e[0][1:]) in ties)]
    if offsets:
        ent += [(f'd{k}', -TIE_DETUNING, TIE_DETUNING, False) for k in sorted(ties)]
    return Space(ent)


def _expand_freqs(P, ties=None):
    """Fill in the tied frequencies: f_dep = f_src (+ d_dep if present)."""
    ties = ties or {}
    return np.stack([P[f'f{ties[k]}'] + P.get(f'd{k}', 0.0) if k in ties else P[f'f{k}']
                     for k in range(N_SITES)], axis=1)


def make_objective_step1(name, prec=P32, space=SPACE_STEP1, ties=None):
    tr = Transition(name)

    def obj(X):
        P = space.decode(X)
        f = _expand_freqs(P, ties)
        c, lev = fidelity_curve(tr, f, COUPLINGS_V4, P['A'], P['sigma'], P['T'], prec=prec, unit=1.0,
                                n_max=N_MAX_CYCLES)
        Fb, nb = best_over_n(c, N_MIN_CYCLES)
        return -Fb, {'F': Fb, 'n': nb, 'A': P['A'], 'sigma': P['sigma'], 'T': P['T'], 'ntr': lev}
    return obj


def load_warm_starts(name, space, folder=None):
    folder = OLD_RESULTS_DIR if folder is None else folder
    xs = []
    for fn in WARM_JSON.get(name, []):
        p = os.path.join(folder, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        prm = {f'f{k}': d['omegas_opt'][k] for k in range(N_SITES)}
        prm.update(A=d['A_opt'], sigma=d.get('sigma_opt', 2.0), T=d['T_opt'])
        xs.append(np.clip(space.encode(prm), 0, 1))
    return xs


def _save_json(out, out_dir, fname, log):
    fn = path_for(fname, out_dir)
    with open(fn, 'w') as fh:
        json.dump(out, fh, indent=2)
    log(f"Saved {fn}")
    return fn


def run_step1(name, use_warm=False, use_ties=True, ftarget=0.995, popsize=32, max_evals=40000,
              max_time=None, polish_evals=20000, seed=1, out_dir=None, log=print):
    """
    Phase A: global search in the space reduced by the resonance constraints (resonance_ties).
    Phase B: refinement in the full space (10 dim) starting from the best point of A.
    """
    log(f"\n================ STEP 1 — transition {name} ================")
    ties, kind = resonance_ties(name) if use_ties else ({}, 'none')
    spA = tied_space(ties, offsets=(kind == 'parallel swaps'))
    warm = load_warm_starts(name, SPACE_STEP1) if use_warm else []
    if ties and warm:     # projection of the warm starts onto the reduced space
        def _proj(w):
            p = {k: v[0] for k, v in SPACE_STEP1.decode(w).items()}
            p.update({f'd{k}': np.clip(p[f'f{k}'] - p[f'f{s}'], -TIE_DETUNING, TIE_DETUNING)
                      for k, s in ties.items()})
            return np.clip(spA.encode(p), 0, 1)
        warm = [_proj(w) for w in warm]
    log(f"Phase A: resonance constraints = {kind} {ties} ({spA.dim} dim)")
    bestA = run_cma(make_objective_step1(name, space=spA, ties=ties), spA.dim, warm_starts=warm,
                    popsize=popsize, max_evals=max_evals, max_time=max_time, ftarget=-ftarget,
                    seed=seed, log=log)
    PA = spA.decode(bestA['x'])
    full = {f'f{k}': _expand_freqs(PA, ties)[0, k] for k in range(N_SITES)}
    full.update(A=PA['A'][0], sigma=PA['sigma'][0], T=PA['T'][0])
    best = dict(bestA, x=SPACE_STEP1.encode(full))
    if polish_evals:
        log("Phase B: refinement in the full space (10 dim)")
        bestB = run_cma(make_objective_step1(name), SPACE_STEP1.dim, warm_starts=[best['x']],
                        popsize=popsize, sigma0_warm=0.003, max_evals=polish_evals, max_restarts=2,
                        local_restarts=False, seed=seed + 1, log=log)
        tot = dict(evals=bestA['evals'] + bestB['evals'], time=bestA['time'] + bestB['time'])
        best = dict(bestB, **tot) if bestB['f'] < best['f'] else dict(best, **tot)
    P = SPACE_STEP1.decode(best['x'])
    f = _expand_freqs(P)[0]
    A, s, T = float(P['A'][0]), float(P['sigma'][0]), float(P['T'][0])
    t = time.perf_counter()
    val = validate(name, f, COUPLINGS_V4, A, s, T, unit=1.0)
    log(f"f64 validation (128 dim, n_tr={val['n_trotter']}): F_pop={val['F_pop']:.6f} "
        f"(F_amp={val['F_amp']:.6f}) at n={val['n_opt']}, Trotter error={val['trotter_err']:.1e} "
        f"({time.perf_counter()-t:.1f}s)")
    tr = Transition(name, reduce_parity=False)
    out = {
        'step': 1, 'transition': name, 'units': 'v4: H=-w/2 sz + J sxsx (rad/ns), t in ns, unit=1',
        'fidelity_definition': 'population |<tgt|psi_n>|^2, max over n in [%d,%d]' % (N_MIN_CYCLES, N_MAX_CYCLES),
        'pulse_shape': 'gaussian',
        'omegas_opt': [float(x) for x in f], 'couplings': [float(x) for x in COUPLINGS_V4],
        'A_opt': A, 'sigma_opt': s, 'T_opt': T, 'lam_equiv': A * math.sqrt(2 * math.pi) * s,
        'n_opt': val['n_opt'], 'fidelity': val['F_pop'], 'fidelity_amp': val['F_amp'],
        'trotter_err': val['trotter_err'], 'n_trotter_val': val['n_trotter'],
        'protocol_time_ns': val['n_opt'] * (T + 6 * s),
        'score_optimizer_f32': -best['f'], 'evals': best['evals'], 'opt_time_s': best['time'],
        'src_index': tr.src_full, 'tgt_index': tr.tgt_full, 'curve': val['curve'],
        'space': SPACE_STEP1.to_json(), 'warm_start': bool(use_warm),
        'resonance_ties_phaseA': {str(k): v for k, v in ties.items()}, 'ties_kind': kind,
    }
    _save_json(out, out_dir, f'v5_step1_{name}.json', log)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. STEP 2 — physical ranges: explicit 2*pi (f, g, A in GHz; t in ns)
# ══════════════════════════════════════════════════════════════════════════════
UNIT_PHYS = 2 * math.pi

# All physical limits in one place (can be changed from the notebook before a run)
PHYS = dict(
    FQ=(4.0, 6.5),        # GHz, qubit frequencies
    DQC=(0.7, 3.0),       # GHz, f_coupler - max(neighbouring qubits): couplers ABOVE the qubits (dispersive)
    FC_MAX=9.0,           # GHz, maximum coupler frequency (sweet spot)
    G=(0.03, 0.15),       # GHz, qubit-coupler couplings
    SIGMA=(2.0, 10.0),    # ns, Gaussian width (referee point (a): >= 2 ns)
    U=(1e-3, 1.0),        # fraction of the maximum allowed excursion (log scale)
    DF_MAX=2.0,           # GHz, maximum coupler excursion 2A (referee point (b))
    F_FLOOR=3.0,          # GHz, the coupler never goes below this frequency during the kick
    KICK_DQC_MIN=0.3,     # GHz, even at the peak of the kick every coupler stays at least this much ABOVE its
                          #      neighbouring qubits: no coupler-qubit crossings (no Landau-Zener)
    T=(10.0, 200.0),      # ns, free time between kicks
    TPROT_MAX=1500.0,     # ns, maximum protocol duration n*(T + 6 sigma)
    ALPHA=-0.2,           # GHz, transmon anharmonicity (qubits and couplers)
    ALPHA_WIN=0.05,       # GHz, forbidden window around ||Delta_qq| - |alpha|| = 0 (|11> <-> |20>)
    PEN_W=0.1,            # penalty weight (full violation = -0.1 in fidelity)
    TIE_D=0.2,            # GHz, detuning allowed in the 'parallel swaps' constraints
)
_QPAIRS = [(0, 2), (2, 4), (4, 6)]          # qubits sharing a coupler


@contextmanager
def phys_override(**kw):
    """Temporarily change PHYS (e.g. TPROT_MAX) and restore it on exit."""
    old = dict(PHYS)
    PHYS.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})
    try:
        yield
    finally:
        PHYS.clear()
        PHYS.update(old)


def step2_space(ties):
    """ties: {dependent: (source, offset_name or None)} on the base names."""
    P = PHYS
    ent = [(f'q{k}', *P['FQ'], False) for k in QUBITS] \
        + [(f'D{c}', 0.0, 1.0, False) for c in COUPLERS] \
        + [(f'g{i}', *P['G'], False) for i in range(N_SITES - 1)] \
        + [('sigma', *P['SIGMA'], False), ('u', *P['U'], True), ('T', *P['T'], False)]
    ent = [e for e in ent if e[0] not in ties]
    ent += [(off, -P['TIE_D'], P['TIE_D'], False) for _, (_, off) in sorted(ties.items()) if off]
    return Space(ent)


def step2_ties(name, use_ties=True):
    """Resonance constraints for phase A (same logic as Step 1, extended to couplers and g)."""
    if not use_ties:
        return {}, 'none'
    src, tgt = TRANSITIONS[name]
    if sorted(N_SITES - 1 - s for s in src) == sorted(tgt):
        t = {'q6': ('q0', None), 'q4': ('q2', None), 'D5': ('D1', None),
             'g5': ('g0', None), 'g4': ('g1', None), 'g3': ('g2', None)}
        return t, 'mirror'
    t = {}
    for s in src:
        for k in tgt:
            if abs(s - k) == 2 and f'q{k}' not in t:
                t[f'q{k}'] = (f'q{s}', f'd{k}')
    return t, 'parallel swaps'


def excursion_max(f):
    """Maximum allowed excursion 2A for frequencies f (B,7) or (7,):
    min(DF_MAX, f_c,min - F_FLOOR, min_c [f_c - max(neighbouring qubits)] - KICK_DQC_MIN)."""
    f = np.atleast_2d(f)
    gap = np.min([f[:, c] - np.maximum(f[:, c - 1], f[:, c + 1]) for c in COUPLERS], axis=0)
    e = np.minimum(PHYS['DF_MAX'], f[:, COUPLERS].min(axis=1) - PHYS['F_FLOOR'])
    return np.maximum(np.minimum(e, gap - PHYS['KICK_DQC_MIN']), 0.0)


def step2_physical(P, ties=None):
    """Optimisation parameters -> physical quantities (GHz, ns) + anharmonicity penalty."""
    ties = ties or {}
    full = dict(P)
    for dep, (src, off) in ties.items():
        full[dep] = full[src] + (full[off] if off else 0.0)
    B = len(full['sigma'])
    lo, hi = PHYS['FQ']
    f = np.zeros((B, N_SITES))
    for k in QUBITS:
        f[:, k] = np.clip(full[f'q{k}'], lo, hi)
    for c in COUPLERS:
        m = np.maximum(f[:, c - 1], f[:, c + 1])
        dmax = np.minimum(PHYS['DQC'][1], PHYS['FC_MAX'] - m)
        f[:, c] = m + PHYS['DQC'][0] + np.clip(full[f'D{c}'], 0, 1) * (dmax - PHYS['DQC'][0])
    g = np.stack([full[f'g{i}'] for i in range(N_SITES - 1)], axis=1)
    A = full['u'] * excursion_max(f) / 2.0   # f_c(t) = f_c - 2 A e^{-t^2/2s^2}
    pen = np.zeros(B)
    for i, j in _QPAIRS:
        x = np.abs(np.abs(f[:, i] - f[:, j]) - abs(PHYS['ALPHA']))
        pen += PHYS['PEN_W'] * np.clip(1.0 - x / PHYS['ALPHA_WIN'], 0, None) ** 2
    return f, g, A, np.asarray(full['sigma']), np.asarray(full['T']), pen


def step2_encode(o):
    """Solution JSON (Step 2/2b/3) -> normalised vector in the full space (inverse of step2_physical)."""
    sp = step2_space({})
    f, g = np.array(o['omegas_opt']), np.array(o['couplings'])
    P = {f'q{k}': f[k] for k in QUBITS}
    for c in COUPLERS:
        m = max(f[c - 1], f[c + 1])
        dmax = min(PHYS['DQC'][1], PHYS['FC_MAX'] - m)
        P[f'D{c}'] = (f[c] - m - PHYS['DQC'][0]) / max(dmax - PHYS['DQC'][0], 1e-12)
    P.update({f'g{i}': g[i] for i in range(N_SITES - 1)})
    exc_max = float(excursion_max(f)[0])
    P.update(sigma=o['sigma_opt'], u=2 * o['A_opt'] / max(exc_max, 1e-12), T=o['T_opt'])
    P = {k: float(np.clip(P[k], lo, hi)) for (k, lo, hi, _) in sp.entries}
    return np.clip(sp.encode(P), 0, 1)


def decode_one(space, x, ties=None):
    """Normalised x -> (f, g, A, sigma, T, pen) of a single candidate."""
    f, g, A, s, T, pen = step2_physical(space.decode(x), ties)
    return f[0], g[0], float(A[0]), float(s[0]), float(T[0]), float(pen[0])


def n_max_allowed(T, sigma):
    """Maximum number of cycles compatible with TPROT_MAX."""
    return np.clip(np.floor(PHYS['TPROT_MAX'] / (np.asarray(T) + 6 * np.asarray(sigma))),
                   0, N_MAX_CYCLES).astype(int)


def make_objective_step2(name, space, ties, model='qubit', prec=P32):
    tr = Transition(name) if model == 'qubit' else qutrit(name)

    def obj(X):
        f, g, A, s, T, pen = step2_physical(space.decode(X), ties)
        c, lev = fidelity_curve(tr, f, g, A, s, T, prec=prec, unit=UNIT_PHYS, n_max=N_MAX_CYCLES)
        Fb, nb = best_over_n(c, N_MIN_CYCLES, n_max_allowed(T, s))
        return -(Fb - pen), {'F': Fb, 'pen': pen, 'n': nb, 'A': A, 'sigma': s, 'T': T, 'ntr': lev}
    return obj


def physics_audit(f, g, A, sigma, T, n_opt):
    """Quantities to report (to the referee) for a solution (GHz, ns)."""
    f = np.asarray(f, float); g = np.asarray(g, float)
    a = abs(PHYS['ALPHA'])
    dqq = [abs(f[i] - f[j]) for i, j in _QPAIRS]
    dqc, gd = [], []
    for c in COUPLERS:
        for q, gi in ((c - 1, g[c - 1]), (c + 1, g[c])):
            dqc.append(f[c] - f[q]); gd.append(gi / (f[c] - f[q]))
    # coupler weight of the single-excitation "qubit-like" eigenstates (proxy for the Purcell effect)
    H1 = np.diag(f) + np.diag(g, 1) + np.diag(g, -1)
    _, evc = np.linalg.eigh(H1)
    part = [float((evc[:, k] ** 2)[COUPLERS].sum()) for k in range(N_SITES)
            if (evc[:, k] ** 2)[QUBITS].sum() > 0.5]
    return {
        'f_qubit_GHz': [float(f[k]) for k in QUBITS],
        'f_coupler_GHz': [float(f[c]) for c in COUPLERS],
        'g_GHz': [float(x) for x in g],
        'min_Delta_qq_GHz': float(min(dqq)),
        'min_dist_from_alpha_resonance_GHz': float(min(abs(d - a) for d in dqq)),
        'min_Delta_qc_GHz': float(min(dqc)),
        'max_g_over_Delta_qc': float(max(gd)),
        'max_coupler_participation_qubit_modes': float(max(part)),
        'coupler_excursion_2A_GHz': float(2 * A),
        'min_coupler_freq_during_kick_GHz': float(min(f[c] for c in COUPLERS) - 2 * A),
        'min_Delta_qc_during_kick_GHz': float(min(dqc) - 2 * A),   # > 0: no coupler-qubit crossing
        'sigma_ns': float(sigma),
        'pulse_bandwidth_GHz': float(1.0 / (2 * math.pi * sigma)),
        'kick_window_ns': float(6 * sigma),
        'period_ns': float(T + 6 * sigma),
        'protocol_time_ns': float(n_opt * (T + 6 * sigma)),
    }


def qutrit_check(name, f, g, A, sigma, T, nmax=4, prec=P64):
    """F(n) curves and leakage into the |2> levels (n = 1..N_MAX_CYCLES) in the three-level transmon model."""
    qt = qutrit(name, nmax)
    UF = floquet_operator(qt, f, g, A, sigma, T, UNIT_PHYS, prec=prec)
    psi = np.zeros(qt.D, complex); psi[qt.src_i] = 1
    Fs, leak = [], []
    for _ in range(N_MAX_CYCLES):
        psi = UF @ psi
        Fs.append(float(abs(psi[qt.tgt_i]) ** 2))
        leak.append(float(np.sum(np.abs(psi[qt.leak_mask]) ** 2)))
    return {'curve': Fs, 'leak': leak, 'D': qt.D}


def pick_n(curve, T, sigma, tol=0.0):
    """n chosen on a curve (respecting N_MIN_CYCLES and TPROT_MAX)."""
    Fb, nb = best_over_n(np.asarray(curve)[None], N_MIN_CYCLES, n_max_allowed(T, sigma), tol=tol)
    return int(nb[0]), float(Fb[0])


def final_evaluation(name, f, g, A, s, T, n_tol=0.0, log=print):
    """
    Full check of a solution: qubit model in f64 (128 dim, Trotter convergence), three-level transmon
    N<=4 (choice of n) and N<=6 (truncation convergence), transmon Trotter error, physics audit.
    """
    val = validate(name, f, g, A, s, T, unit=UNIT_PHYS, n_max=N_MAX_CYCLES)
    Q4 = qutrit_check(name, f, g, A, s, T, nmax=4)
    n, _ = pick_n(Q4['curve'], T, s, n_tol)
    Q6 = qutrit_check(name, f, g, A, s, T, nmax=6, prec=P32)
    c2, _ = fidelity_curve(qutrit(name), np.atleast_2d(f), g, [A], [s], [T], prec=P64, unit=UNIT_PHYS,
                           n_max=n, n_tr=n_trotter(A, s, UNIT_PHYS, 4))
    R = {'n': n, 'F': Q4['curve'][n - 1], 'leak': Q4['leak'][n - 1],
         'F_N6': Q6['curve'][n - 1], 'leak_N6': Q6['leak'][n - 1],
         'F_qubit': val['curve'][n - 1], 'F_qubit_best': val['F_pop'], 'n_qubit': val['n_opt'],
         'trotter_err_qubit': val['trotter_err'], 'trotter_err_qutrit': abs(float(c2[0, n - 1]) - Q4['curve'][n - 1]),
         'curve_qubit': val['curve'], 'curve_qutrit_N4': Q4['curve'], 'D4': Q4['D'], 'D6': Q6['D'],
         'audit': physics_audit(f, g, A, s, T, n)}
    log(f"Check: transmon N<=4 (D={Q4['D']}) F={R['F']:.6f} at n={n} (|2> leakage {R['leak']:.1e}); "
        f"N<=6 (D={Q6['D']}) F={R['F_N6']:.6f}; qubit model f64 at the same n {R['F_qubit']:.6f}; Trotter error "
        f"qubit {R['trotter_err_qubit']:.1e}, transmon {R['trotter_err_qutrit']:.1e}")
    return R


def log_audit(audit, log):
    log("Physics audit:")
    for k, v in audit.items():
        log(f"   {k:40s} = {np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)}")


def solution_json(name, step, f, g, A, s, T, R, extra=None):
    """Common JSON dictionary for solutions (Step 2, 2b, 3-opt): keys compatible with the notebook."""
    tr = Transition(name, reduce_parity=False)
    out = {
        'step': step, 'transition': name,
        'units': 'GHz and ns: H = 2*pi*[ -f_i/2 sz_i + g_i sx sx + A e^{-t^2/2s^2} sum_c sz_c ]',
        'fidelity_definition': 'population |<tgt|psi_n>|^2 in the three-level transmon model (N<=4), '
                               'n in [%d, min(%d, TPROT_MAX/(T+6s))]' % (N_MIN_CYCLES, N_MAX_CYCLES),
        'pulse_shape': 'gaussian', 'model_final': 'qutrit',
        'omegas_opt': [float(x) for x in f], 'couplings': [float(x) for x in g],
        'A_opt': float(A), 'sigma_opt': float(s), 'T_opt': float(T), 'n_opt': int(R['n']),
        'fidelity': R['F'],
        'fidelity_qubit_at_n_opt': R['F_qubit'], 'fidelity_qubit_best': R['F_qubit_best'],
        'n_opt_qubit': R['n_qubit'],
        'fidelity_qutrit_N4': R['F'], 'leakage_qutrit_N4': R['leak'],
        'fidelity_qutrit_N6': R['F_N6'], 'leakage_qutrit_N6': R['leak_N6'],
        'trotter_err': R['trotter_err_qubit'], 'trotter_err_qutrit': R['trotter_err_qutrit'],
        'protocol_time_ns': R['audit']['protocol_time_ns'], 'audit': R['audit'], 'PHYS': dict(PHYS),
        'src_index': tr.src_full, 'tgt_index': tr.tgt_full,
        'curve_qubit': R['curve_qubit'], 'curve_qutrit_N4': R['curve_qutrit_N4'],
    }
    out.update(extra or {})
    return out


def run_step2(name, use_ties=True, ftarget=0.995, popsize=32, max_evals=300000, max_time=1500,
              polish_evals=20000, qutrit_refine_tol=5e-3, qutrit_refine_evals=6000,
              auto_refine=True, refine_target=0.99, refine_evals=40000, refine_time=3000,
              seed=1, out_dir=None, log=print):
    """
    Phase A: global search with resonance constraints (qubit model, GHz/ns with 2*pi).
    Phase B: refinement in the full space (16 parameters).
    Phase C (if |F_qubit - F_transmon| > qutrit_refine_tol): refinement with the transmon model in the objective.
    Validation: final_evaluation. If auto_refine and the final (transmon) F is < refine_target, at the end
    refine_step2_qutrit is called (long refinement in the three-level model).
    """
    log(f"\n================ STEP 2 — transition {name} (GHz, ns, explicit 2*pi) ================")
    log("Physical limits: " + ", ".join(f"{k}={v}" for k, v in PHYS.items()))
    ties, kind = step2_ties(name, use_ties)
    spA, spF = step2_space(ties), step2_space({})
    log(f"Phase A: resonance constraints = {kind} ({spA.dim} dim)")
    bA = run_cma(make_objective_step2(name, spA, ties), spA.dim, popsize=popsize,
                 max_evals=max_evals, max_time=max_time, ftarget=-ftarget, seed=seed, log=log)
    x_best, f_best = _to_full(spA, spF, ties, bA['x']), bA['f']
    evals, ttot = bA['evals'], bA['time']
    if polish_evals:
        log(f"Phase B: refinement in the full space ({spF.dim} dim)")
        bB = run_cma(make_objective_step2(name, spF, {}), spF.dim, warm_starts=[x_best], popsize=popsize,
                     sigma0_warm=0.003, max_evals=polish_evals, max_restarts=2, local_restarts=False,
                     seed=seed + 1, log=log)
        evals += bB['evals']; ttot += bB['time']
        if bB['f'] < f_best:
            x_best, f_best = bB['x'], bB['f']
    model_final = 'qubit'
    f, g, A, s, T, pen = decode_one(spF, x_best)
    R = final_evaluation(name, f, g, A, s, T, log=log)
    if abs(R['F_qubit'] - R['F']) > qutrit_refine_tol and qutrit_refine_evals:
        log(f"Phase C: |F_qubit - F_transmon| > {qutrit_refine_tol} -> refinement with the transmon model in the objective")
        bC = run_cma(make_objective_step2(name, spF, {}, model='qutrit'), spF.dim, warm_starts=[x_best],
                     popsize=16, sigma0_warm=0.003, max_evals=qutrit_refine_evals, max_restarts=3,
                     local_restarts=True, seed=seed + 2, log=log)
        evals += bC['evals']; ttot += bC['time']
        x_best, model_final = bC['x'], 'qutrit'
        f, g, A, s, T, pen = decode_one(spF, x_best)
        R = final_evaluation(name, f, g, A, s, T, log=log)
    log_audit(R['audit'], log)
    out = solution_json(name, 2, f, g, A, s, T, R, {
        'model_final': model_final, 'penalty': pen, 'evals': evals, 'opt_time_s': ttot, 'ties_kind': kind})
    _save_json(out, out_dir, f'v5_step2_{name}.json', log)
    if auto_refine and out['fidelity'] < refine_target:
        log(f"F in the three-level model = {out['fidelity']:.5f} < {refine_target}: automatic transmon refinement "
            f"(up to {refine_time} s)")
        out = refine_step2_qutrit(name, max_evals=refine_evals, max_time=refine_time, out_dir=out_dir, log=log)
    return out


def _to_full(spA, spF, ties, x):
    """Vector of the constrained space -> vector of the full space."""
    P = {k: v[0] for k, v in spA.decode(x).items()}
    for dep, (src, off) in ties.items():
        P[dep] = P[src] + (P[off] if off else 0.0)
    return spF.encode({k: float(np.clip(P[k], lo, hi)) for (k, lo, hi, _) in spF.entries})


def refine_step2_qutrit(name, max_evals=40000, max_time=3600, popsize=16, ftarget=0.995,
                        seed=11, out_dir=None, log=print):
    """
    Reload v5_step2_<name>.json and continue the optimisation DIRECTLY in the three-level transmon
    model (N<=4), with alternating local and random restarts.
    The JSON is overwritten only if the transmon fidelity improves.
    """
    fn = path_for(f'v5_step2_{name}.json', out_dir)
    o = json.load(open(fn))
    F_old = o['fidelity_qutrit_N4']
    log(f"\n======== STEP 2 — transmon refinement of {name} (current F_transmon={F_old:.5f}) ========")
    spF = step2_space({})
    b = run_cma(make_objective_step2(name, spF, {}, model='qutrit'), spF.dim, warm_starts=[step2_encode(o)],
                popsize=popsize, sigma0=0.15, sigma0_warm=0.01, max_evals=max_evals, max_time=max_time,
                ftarget=-ftarget, seed=seed, local_restarts=True, log=log)
    f, g, A, s, T, pen = decode_one(spF, b['x'])
    R = final_evaluation(name, f, g, A, s, T, log=log)
    if R['F'] <= F_old:
        log(f"No improvement over {F_old:.5f}: JSON left unchanged")
        return o
    log_audit(R['audit'], log)
    out = solution_json(name, 2, f, g, A, s, T, R, {
        'penalty': pen, 'evals': o.get('evals', 0) + b['evals'], 'opt_time_s': o.get('opt_time_s', 0) + b['time'],
        'ties_kind': o.get('ties_kind')})
    _save_json(out, out_dir, f'v5_step2_{name}.json', log)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. STEP 2b — transfer PRODUCED BY THE Floquet DRIVE, gate <= 450 ns
#   * penalty if the transfer also happens WITHOUT the pulse:
#       F_static(t <= STATIC_TMAX) <= F_STATIC_MAX  (otherwise -PEN_STATIC_W * excess)
#   * the FIRST n within N_EARLY_TOL of the maximum is chosen (shorter gate for the same F)
# ══════════════════════════════════════════════════════════════════════════════
STEP2B = dict(
    TPROT_MAX=450.0,       # ns, maximum gate duration
    STATIC_TMAX=1500.0,    # ns, window in which the transfer without pulse is checked
    STATIC_DT=0.5,         # ns, time step of the static check inside the objective
    F_STATIC_MAX=0.3,      # tolerated F without pulse
    PEN_STATIC_W=1.0,      # penalty = W * max(0, F_static - F_STATIC_MAX)
    N_EARLY_TOL=5e-4,      # the first n with F >= F_max - tol is chosen
)


def static_max_batch(tr, f, g, t_max=None, dt=None, t_chunk=512, return_time=False):
    """max_t |<tgt|e^{-i H0 t}|src>|^2 for a batch (A = 0), t in [0, t_max]. GPU if available."""
    t_max = STEP2B['STATIC_TMAX'] if t_max is None else t_max
    dt = STEP2B['STATIC_DT'] if dt is None else dt
    E, V = np.linalg.eigh(build_H0(tr, f, g, UNIT_PHYS))
    w = V[:, tr.tgt_i, :] * V[:, tr.src_i, :]
    Et = torch.as_tensor(E, dtype=torch.float64, device=DEVICE)
    wt = torch.as_tensor(w, dtype=torch.complex128, device=DEVICE)
    t = torch.arange(0.0, t_max + dt / 2, dt, dtype=torch.float64, device=DEVICE)
    Fm = torch.zeros(len(E), dtype=torch.float64, device=DEVICE)
    tm = torch.zeros(len(E), dtype=torch.float64, device=DEVICE)
    for j in range(0, len(t), t_chunk):
        amp = (_cis(-Et[:, None, :] * t[None, j:j + t_chunk, None]) * wt[:, None, :]).sum(-1)
        Fc, jc = (amp.abs() ** 2).max(dim=1)
        better = Fc > Fm
        Fm = torch.where(better, Fc, Fm)
        tm = torch.where(better, t[j:j + t_chunk][jc], tm)
    if return_time:
        return Fm.cpu().numpy(), tm.cpu().numpy()
    return Fm.cpu().numpy()


def static_fidelity(name, f, g, t_max=None, dt=0.1, nmax=4):
    """Maximum F_pop reachable WITHOUT the pulse (A = 0), for t in [0, t_max] (transmon model)."""
    t_max = PHYS['TPROT_MAX'] if t_max is None else t_max
    F, t = static_max_batch(qutrit(name, nmax), np.atleast_2d(f), np.asarray(g), t_max, dt, return_time=True)
    return {'F_static_max': float(F[0]), 't_static_ns': float(t[0])}


def static_penalty(Fs):
    return STEP2B['PEN_STATIC_W'] * np.clip(np.asarray(Fs) - STEP2B['F_STATIC_MAX'], 0, None)


def make_objective_step2b(name, space, ties, model='qubit'):
    tr = Transition(name) if model == 'qubit' else qutrit(name)

    def obj(X):
        f, g, A, s, T, pen = step2_physical(space.decode(X), ties)
        c, lev = fidelity_curve(tr, f, g, A, s, T, prec=P32, unit=UNIT_PHYS, n_max=N_MAX_CYCLES)
        Fb, nb = best_over_n(c, N_MIN_CYCLES, n_max_allowed(T, s), tol=STEP2B['N_EARLY_TOL'])
        Fs = static_max_batch(tr, f, g)
        return -(Fb - pen - static_penalty(Fs)), {'F': Fb, 'Fstat': Fs, 'pen': pen, 'n': nb, 'A': A,
                                                  'sigma': s, 'T': T}
    return obj


def floquet_extra(name, f, g, F):
    """Static contribution (no pulse) within the gate time and within STATIC_TMAX, and the Step 2b score."""
    st_gate = static_fidelity(name, f, g, t_max=PHYS['TPROT_MAX'])
    st_long = static_fidelity(name, f, g, t_max=STEP2B['STATIC_TMAX'])
    return {'F_static_gate_window': st_gate['F_static_max'], 'F_static_long_window': st_long['F_static_max'],
            't_static_long_ns': st_long['t_static_ns'], 'floquet_contribution': F - st_long['F_static_max'],
            'score_2b': F - float(static_penalty(st_long['F_static_max']))}


def run_step2b(name, use_ties=True, ftarget=0.995, popsize=32, max_evals=300000, max_time=1800,
               polish_evals=20000, polish_time=None, model_search='qutrit', qutrit_evals=0, qutrit_time=1200,
               warm_from=(), seed=1, out_dir=None, fresh=False, log=print):
    """
    Phase A: global search (resonance constraints if use_ties) with static penalty and gate <= TPROT_MAX.
    Phase B: refinement in the full space. Both in the `model_search` model:
             'qutrit' (reliable: with large kicks the qubit model fails), 'qubit' (fast, exploratory).
    Phase C (only if model_search='qubit' and qutrit_evals > 0): refinement in the three-level model.
    warm_from: JSONs of other solutions used as warm starts (only with use_ties=False).
    One checkpoint per phase in checkpoints/2b_<name>_*.pkl: an interrupted run resumes where it stopped.
    fresh=True deletes the checkpoints and starts from scratch. Saves v5_step2b_<name>.json.
    """
    if fresh:
        clear_checkpoints(out_dir, '2b', name)
    with phys_override(TPROT_MAX=STEP2B['TPROT_MAX']):
        log(f"\n================ STEP 2b — {name}: drive-essential transfer, gate <= {PHYS['TPROT_MAX']:.0f} ns ================")
        log("Physical limits: " + ", ".join(f"{k}={v}" for k, v in PHYS.items()))
        log("Step 2b: " + ", ".join(f"{k}={v}" for k, v in STEP2B.items()))
        ties, kind = step2_ties(name, use_ties)
        spA, spF = step2_space(ties), step2_space({})
        warm = [step2_encode(json.load(open(path_for(p, out_dir)))) for p in warm_from] if not ties else []
        if warm_from:
            log(f"Warm starts from: {list(warm_from)}" + ("" if not ties else " (IGNORED: use_ties=False is required)"))
        log(f"Phase A: constraints = {kind} ({spA.dim} dim), model {model_search}, static penalty")
        bA = run_cma(make_objective_step2b(name, spA, ties, model=model_search), spA.dim, warm_starts=warm,
                     sigma0_warm=0.08, popsize=popsize, max_evals=max_evals, max_time=max_time,
                     ftarget=-ftarget, seed=seed, log=log, checkpoint=ckpt_path(out_dir, '2b', name, 'A'),
                     sig_extra=dict(model=model_search, ties=kind, STEP2B=dict(STEP2B), warm=list(warm_from)))
        x_best, f_best = _to_full(spA, spF, ties, bA['x']), bA['f']
        evals, ttot = bA['evals'], bA['time']
        if polish_evals:
            log(f"Phase B: refinement in the full space ({spF.dim} dim), model {model_search}")
            bB = run_cma(make_objective_step2b(name, spF, {}, model=model_search), spF.dim, warm_starts=[x_best],
                         popsize=popsize, sigma0_warm=0.003, max_evals=polish_evals, max_time=polish_time,
                         max_restarts=2, local_restarts=False, seed=seed + 1, log=log,
                         checkpoint=ckpt_path(out_dir, '2b', name, 'B'),
                         sig_extra=dict(model=model_search, STEP2B=dict(STEP2B), x0=np.round(x_best, 10).tolist()))
            evals += bB['evals']; ttot += bB['time']
            if bB['f'] < f_best:
                x_best, f_best = bB['x'], bB['f']
        if model_search == 'qubit' and qutrit_evals:
            log("Phase C: refinement in the three-level model")
            bC = run_cma(make_objective_step2b(name, spF, {}, model='qutrit'), spF.dim, warm_starts=[x_best],
                         popsize=12, sigma0=0.1, sigma0_warm=0.01, max_evals=qutrit_evals, max_time=qutrit_time,
                         ftarget=-ftarget, max_restarts=20, local_restarts=True, seed=seed + 2, log=log,
                         checkpoint=ckpt_path(out_dir, '2b', name, 'C'),
                         sig_extra=dict(STEP2B=dict(STEP2B), x0=np.round(x_best, 10).tolist()))
            evals += bC['evals']; ttot += bC['time']
            x_best = bC['x']
        f, g, A, s, T, pen = decode_one(spF, x_best)
        R = final_evaluation(name, f, g, A, s, T, n_tol=STEP2B['N_EARLY_TOL'], log=log)
        fx = floquet_extra(name, f, g, R['F'])
        log(f"Without pulse: max F {fx['F_static_gate_window']:.4f} within {PHYS['TPROT_MAX']:.0f} ns, "
            f"{fx['F_static_long_window']:.4f} within {STEP2B['STATIC_TMAX']:.0f} ns -> Floquet contribution "
            f"C = {fx['floquet_contribution']:.4f}")
        log_audit(R['audit'], log)
        out = solution_json(name, '2b', f, g, A, s, T, R, dict(
            fx, penalty=pen, STEP2B=dict(STEP2B), model_search=model_search, evals=evals, opt_time_s=ttot,
            ties_kind=kind, warm_from=list(warm_from)))
    _save_json(out, out_dir, f'v5_step2b_{name}.json', log)
    clear_checkpoints(out_dir, '2b', name)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 7. STEP 3 — analysis: Floquet contribution, decoherence (Lindblad), quasi-static noise, recalibration
#    Everything in the three-level transmon model (alpha = PHYS['ALPHA']), GHz and ns.
# ══════════════════════════════════════════════════════════════════════════════
NOISE = dict(
    T1_Q=50e3, TPHI_Q=80e3,        # ns: qubits (T1 = 50 us, Tphi = 80 us)
    T1_C=10e3, TPHI_C=20e3,        # ns: couplers (flux-tunable: worse coherence)
    SEG_MAX=5.0,                   # ns: maximum free segment in the unitary/dissipative splitting
    DF_Q=0.2e-3, DF_C=0.5e-3,      # GHz: ABSOLUTE quasi-static noise (standard deviation) on qubits and couplers
    DG_REL=5e-3, DA_REL=5e-3,      # relative: on g_i and on the amplitude A (AWG)         [quasi_static_noise]
    N_SAMPLES=256,                 # samples per noise level                              [quasi_static_noise]
    SCALES=(0.1, 0.3, 1.0, 3.0, 10.0),      # multipliers of the absolute noise
    T1_SCALES=(0.2, 0.5, 1.0, 2.0, 5.0),    # multipliers of T1 and Tphi (F vs T1 curve)
    SIGMA_FAB=(0.002, 0.005, 0.02),         # GHz: fabrication spread of the qubits
    N_CHIPS=6, RECAL_EVALS=1000,            # simulated chips per sigma_fab, evaluations per recalibration
)

# RELATIVE quasi-static noise (same model as the noise scan of the original paper):
#   f_i -> f_i (1 + eta Z_i),  g_i -> g_i (1 + eta Z'_i);  A, sigma, T, n noise-free (control parameters)
ETA = dict(
    ETA_LIST=(0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    N_SAMPLES=128,
    ON_A=False,            # True: also A -> A (1 + eta Z) (AWG noise)
)

_COMB = np.array([[1, 0, 0], [1, 1, 0], [1, 2, 1]], dtype=float)   # C(n, k) for n, k <= 2


def _sol(o):
    return (np.asarray(o['omegas_opt'], float), np.asarray(o['couplings'], float),
            float(o['A_opt']), float(o['sigma_opt']), float(o['T_opt']), int(o['n_opt']))


def floquet_contribution(name, o):
    """C = F(with pulse) - max F(without pulse, t <= max(TPROT_MAX, STATIC_TMAX)): how much the drive matters."""
    f, g, *_ = _sol(o)
    t_win = max(PHYS['TPROT_MAX'], STEP2B['STATIC_TMAX'])
    s = static_fidelity(name, f, g, t_max=t_win)
    s['t_window_ns'] = t_win
    s['F_static_gate'] = static_fidelity(name, f, g, t_max=PHYS['TPROT_MAX'])['F_static_max']
    s['F_drive'] = float(o['fidelity'])
    s['floquet_contribution'] = s['F_drive'] - s['F_static_max']
    return s


class OpenQutrit:
    """
    Three-level transmons, BOTH parities (N_tot <= nmax), for the Lindblad equation
        drho/dt = -i[H(t), rho] + sum_i D[sqrt(1/T1_i) a_i] rho + sum_i D[sqrt(2/Tphi_i) n_i] rho
    Strang splitting between the unitary part (exact: kick + free segments) and EXACT local channels
    (bosonic amplitude damping truncated to 3 levels; pure dephasing).
    """

    def __init__(self, name, nmax=4):
        qt = qutrit(name, nmax, both_parities=True)
        self.qt, self.D, self.name = qt, qt.D, name
        pos = {s: j for j, s in enumerate(qt.states)}
        self.n = np.array(qt.states).T.astype(float)                 # (7, D)
        self.lower = {}
        for i in range(N_SITES):
            for k in (1, 2):
                src = np.array([j for j, s in enumerate(qt.states) if s[i] >= k])
                dst = np.array([pos[qt.states[j][:i] + (qt.states[j][i] - k,) + qt.states[j][i + 1:]]
                                for j in src])
                self.lower[(i, k)] = (src, dst)
        self._deph_cache = {}

    def rates(self, scale=1.0):
        T1 = np.array([NOISE['T1_C'] if k in COUPLERS else NOISE['T1_Q'] for k in range(N_SITES)]) * scale
        Tp = np.array([NOISE['TPHI_C'] if k in COUPLERS else NOISE['TPHI_Q'] for k in range(N_SITES)]) * scale
        return T1, Tp

    def dissipate(self, rho, tau, T1, Tphi):
        if tau <= 0:
            return rho
        key = (round(tau, 12), tuple(Tphi))
        if key not in self._deph_cache:
            M = np.zeros((self.D, self.D))
            for i in range(N_SITES):
                if np.isfinite(Tphi[i]):
                    d = self.n[i][:, None] - self.n[i][None, :]
                    M += d ** 2 / Tphi[i]
            self._deph_cache[key] = np.exp(-tau * M)
        rho = rho * self._deph_cache[key]
        for i in range(N_SITES):
            if not np.isfinite(T1[i]):
                continue
            p = 1.0 - math.exp(-tau / T1[i])
            ni = self.n[i]
            k0 = (1.0 - p) ** (ni / 2.0)
            new = rho * np.outer(k0, k0)
            for k in (1, 2):
                src, dst = self.lower[(i, k)]
                nn = ni[src].astype(int)
                w = np.sqrt(_COMB[nn, k] * p ** k * (1.0 - p) ** (nn - k))
                new[np.ix_(dst, dst)] += rho[np.ix_(src, src)] * np.outer(w, w)
            rho = new
        return rho

    def propagators(self, f, g, A, sigma, T, seg_max):
        ev, evc = diagonalize(build_H0(self.qt, f[None], g, UNIT_PHYS))
        Uk = kick_unitary(self.qt, ev, evc, _t64(A), _t64(sigma), n_trotter(A, sigma, UNIT_PHYS, 2), P64,
                          UNIT_PHYS)[0].cpu().numpy()
        E, V = ev[0].cpu().numpy(), evc[0].cpu().numpy()
        n_seg = max(1, int(math.ceil(T / seg_max)))
        tau = T / n_seg
        Useg = (V * np.exp(-1j * E * tau)) @ V.T
        return Uk, Useg, n_seg, tau

    def evolve(self, o, n_cycles=None, T1=None, Tphi=None, seg_max=None, f=None, g=None):
        """F(n), |2> leakage, population lost from the two-excitation manifold, n = 1..n_cycles."""
        f0, g0, A, s, T, n_opt = _sol(o)
        f = f0 if f is None else f
        g = g0 if g is None else g
        n_cycles = N_MAX_CYCLES if n_cycles is None else n_cycles
        seg_max = NOISE['SEG_MAX'] if seg_max is None else seg_max
        if T1 is None:
            T1, Tphi = self.rates()
        Uk, Useg, n_seg, tau = self.propagators(f, g, A, s, T, seg_max)
        Ukh, Usegh = Uk.conj().T, Useg.conj().T
        tk = 2 * WINDOW * s
        qt = self.qt
        rho = np.zeros((self.D, self.D), complex)
        rho[qt.src_i, qt.src_i] = 1.0
        N_tot = self.n.sum(axis=0)
        F, leak, loss, tr_err = [], [], [], 0.0
        for _ in range(n_cycles):
            rho = self.dissipate(rho, tk / 2, T1, Tphi)
            rho = Uk @ rho @ Ukh
            rho = self.dissipate(rho, (tk + tau) / 2, T1, Tphi)
            for m in range(n_seg):
                rho = Useg @ rho @ Usegh
                rho = self.dissipate(rho, tau if m < n_seg - 1 else tau / 2, T1, Tphi)
            p = np.real(np.diag(rho))
            F.append(float(p[qt.tgt_i]))
            leak.append(float(p[qt.leak_mask].sum()))
            loss.append(float(p[N_tot < 2].sum()))
            tr_err = max(tr_err, abs(np.trace(rho).real - 1.0))
        min_eig = float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min())
        return {'F': F, 'leak': leak, 'loss_to_N<2': loss, 'trace_err': tr_err, 'min_eig_final': min_eig,
                'n_seg': n_seg}


def lindblad_analysis(name, o, log=print, T1_scan=True):
    """F with decoherence at n_opt; checks (T1 -> inf, splitting convergence); F vs T1 curve."""
    oq = OpenQutrit(name)
    *_, n_opt = _sol(o)
    t = time.perf_counter()
    inf = np.full(N_SITES, np.inf)
    U = oq.evolve(o, n_cycles=n_opt, T1=inf, Tphi=inf)
    R = oq.evolve(o)
    R2 = oq.evolve(o, n_cycles=n_opt, seg_max=NOISE['SEG_MAX'] / 2)
    nmax_ok = int(n_max_allowed(o['T_opt'], o['sigma_opt']))
    out = {
        'D_open': oq.D,
        'F_unitary_check': U['F'][n_opt - 1],
        'F_lindblad': R['F'][n_opt - 1],
        'F_lindblad_best_n': max(R['F'][N_MIN_CYCLES - 1:max(nmax_ok, N_MIN_CYCLES)]),
        'leak_lindblad': R['leak'][n_opt - 1], 'loss_to_N<2': R['loss_to_N<2'][n_opt - 1],
        'splitting_err': abs(R2['F'][n_opt - 1] - R['F'][n_opt - 1]),
        'trace_err': R['trace_err'], 'min_eig_final': R['min_eig_final'],
        'curve_lindblad': R['F'],
        'rates_ns': {k: NOISE[k] for k in ('T1_Q', 'TPHI_Q', 'T1_C', 'TPHI_C')},
    }
    log(f"  Lindblad (D={oq.D}): unitary F {out['F_unitary_check']:.5f} (JSON {o['fidelity']:.5f}) -> "
        f"with decoherence {out['F_lindblad']:.5f} at n={n_opt}; loss to N<2 {out['loss_to_N<2']:.1e}, "
        f"leakage {out['leak_lindblad']:.1e}")
    log(f"  checks: splitting error {out['splitting_err']:.1e}, |tr-1| {out['trace_err']:.1e}, "
        f"min eigenvalue {out['min_eig_final']:.1e} ({time.perf_counter()-t:.0f}s)")
    if T1_scan:
        scan = []
        for sc in NOISE['T1_SCALES']:
            T1, Tp = oq.rates(sc)
            r = oq.evolve(o, n_cycles=n_opt, T1=T1, Tphi=Tp)
            scan.append({'scale': sc, 'T1_qubit_us': NOISE['T1_Q'] * sc / 1e3, 'F': r['F'][n_opt - 1]})
            log(f"    T1/Tphi x{sc:<4}: T1_q = {NOISE['T1_Q']*sc/1e3:5.0f} us -> F = {r['F'][n_opt-1]:.5f}")
        out['T1_scan'] = scan
    return out


def eta_scan(name, o, eta_list=None, n_samples=None, seed=0, on_A=None, batch=32, log=print):
    """
    Relative quasi-static noise. For every eta: F at FIXED n = n_opt (what is done in the lab)
    and the max of F over n (metric of the original code). Same Z samples for all eta. Transmon model.
    Also returns the mean and std F(n) curves for the band plots.
    """
    eta_list = ETA['ETA_LIST'] if eta_list is None else eta_list
    n_samples = ETA['N_SAMPLES'] if n_samples is None else n_samples
    on_A = ETA['ON_A'] if on_A is None else on_A
    f0, g0, A0, s, T, n_opt = _sol(o)
    qt = qutrit(name)
    rng = np.random.default_rng(seed)
    Zf, Zg, Za = rng.normal(size=(n_samples, N_SITES)), rng.normal(size=(n_samples, N_SITES - 1)), \
        rng.normal(size=n_samples)
    nmax_ok = max(int(n_max_allowed(T, s)), N_MIN_CYCLES)
    fq_mean = float(np.mean(f0[QUBITS]))
    res = []
    for eta in eta_list:
        A = A0 * (1 + eta * Za) if on_A else A0
        C = fidelity_curve_batched(qt, f0[None] * (1 + eta * Zf), g0[None] * (1 + eta * Zg), A, s, T,
                                   batch=batch, prec=P32, unit=UNIT_PHYS, n_max=N_MAX_CYCLES)
        Ffix = C[:, n_opt - 1]
        Fmx, _ = best_over_n(C, N_MIN_CYCLES, np.full(n_samples, nmax_ok))
        q = np.sort(Ffix)
        r = {'eta': eta, 'df_qubit_MHz': eta * fq_mean * 1e3,
             'mean': float(Ffix.mean()), 'std': float(Ffix.std()), 'p20': float(np.percentile(Ffix, 20)),
             'cvar20': float(q[:max(1, n_samples // 5)].mean()), 'min': float(Ffix.min()),
             'mean_maxn': float(Fmx.mean()), 'std_maxn': float(Fmx.std()),
             'curve_mean': C.mean(0).tolist(), 'curve_std': C.std(0).tolist()}
        res.append(r)
        log(f"    eta = {eta:7.1e} (df_q ~ {r['df_qubit_MHz']:6.3f} MHz): fixed n: mean {r['mean']:.5f} "
            f"+- {r['std']:.5f}, p20 {r['p20']:.5f}, min {r['min']:.5f} | max over n: {r['mean_maxn']:.5f} "
            f"+- {r['std_maxn']:.5f}")
    return {'levels': res, 'n_opt': n_opt, 'n_max_allowed': nmax_ok, 'on_A': on_A,
            'model': 'f_i(1+eta Z), g_i(1+eta Z), A/T/sigma noise-free' + (' (noisy A)' if on_A else '')}


def quasi_static_noise(name, o, scales=None, n_samples=None, seed=0, log=print, batch=32):
    """ABSOLUTE static noise (NOISE: DF_Q, DF_C, DG_REL, DA_REL) at FIXED n = n_opt."""
    scales = NOISE['SCALES'] if scales is None else scales
    n_samples = NOISE['N_SAMPLES'] if n_samples is None else n_samples
    f0, g0, A0, s, T, n_opt = _sol(o)
    qt = qutrit(name)
    rng = np.random.default_rng(seed)
    Zf, Zg, Za = rng.normal(size=(n_samples, N_SITES)), rng.normal(size=(n_samples, N_SITES - 1)), \
        rng.normal(size=n_samples)
    dfs = np.array([NOISE['DF_C'] if k in COUPLERS else NOISE['DF_Q'] for k in range(N_SITES)])
    res = []
    for sc in scales:
        Fs = fidelity_curve_batched(qt, f0[None] + sc * Zf * dfs, g0[None] * (1 + sc * NOISE['DG_REL'] * Zg),
                                    A0 * (1 + sc * NOISE['DA_REL'] * Za), s, T, batch=batch, prec=P32,
                                    unit=UNIT_PHYS, n_max=n_opt)[:, n_opt - 1]
        q = np.sort(Fs)
        r = {'scale': sc, 'df_qubit_MHz': NOISE['DF_Q'] * sc * 1e3, 'mean': float(Fs.mean()),
             'std': float(Fs.std()), 'p20': float(np.percentile(Fs, 20)),
             'cvar20': float(q[:max(1, n_samples // 5)].mean()), 'min': float(Fs.min())}
        res.append(r)
        log(f"    noise x{sc:<4} (df_q = {r['df_qubit_MHz']:.2f} MHz, dA/A = {NOISE['DA_REL']*sc:.1e}): "
            f"mean {r['mean']:.5f}  p20 {r['p20']:.5f}  min {r['min']:.5f}")
    return {'levels': res, 'base': {k: NOISE[k] for k in ('DF_Q', 'DF_C', 'DG_REL', 'DA_REL')}}


def recalibration_test(name, o, sigma_fab=None, n_chips=None, max_evals=None, seed=0, log=print):
    """
    'Real' chip: qubit frequencies shifted by N(0, sigma_fab) (fabrication, cannot be corrected).
    ONLY the pulse parameters (sigma, A via u, T) and n are re-optimised.
    """
    sigma_fab = NOISE['SIGMA_FAB'] if sigma_fab is None else sigma_fab
    n_chips = NOISE['N_CHIPS'] if n_chips is None else n_chips
    max_evals = NOISE['RECAL_EVALS'] if max_evals is None else max_evals
    f0, g0, A0, s0, T0, n_opt = _sol(o)
    qt = qutrit(name)
    sp = Space([('sigma', *PHYS['SIGMA'], False), ('u', *PHYS['U'], True), ('T', *PHYS['T'], False)])
    rng = np.random.default_rng(seed)
    out = []
    for sf in sigma_fab:
        rows = []
        for chip in range(n_chips):
            f = f0.copy()
            f[QUBITS] += rng.normal(0, sf, len(QUBITS))
            em = float(excursion_max(f)[0])
            c, _ = fidelity_curve(qt, f[None], g0, A0, s0, T0, prec=P32, unit=UNIT_PHYS, n_max=n_opt)
            F_before = float(c[0, n_opt - 1])

            def obj(X, f=f, em=em):
                P = sp.decode(X)
                cc, _ = fidelity_curve(qt, np.broadcast_to(f, (len(X), N_SITES)), g0, P['u'] * em / 2.0,
                                       P['sigma'], P['T'], prec=P32, unit=UNIT_PHYS, n_max=N_MAX_CYCLES)
                Fb, nb = best_over_n(cc, N_MIN_CYCLES, n_max_allowed(P['T'], P['sigma']))
                return -Fb, {'F': Fb, 'n': nb}

            u0 = float(np.clip(2 * A0 / max(em, 1e-12), *PHYS['U']))
            b = run_cma(obj, sp.dim, warm_starts=[sp.encode({'sigma': s0, 'u': u0, 'T': T0})], popsize=12,
                        sigma0=0.2, sigma0_warm=0.05, max_evals=max_evals, max_restarts=6, ftarget=-0.995,
                        seed=seed + chip, local_restarts=True, log=lambda *_: None)
            rows.append({'F_before': F_before, 'F_after': -b['f'], 'df_MHz': ((f - f0)[QUBITS] * 1e3).tolist()})
        Fb = np.array([r['F_before'] for r in rows]); Fa = np.array([r['F_after'] for r in rows])
        out.append({'sigma_fab_MHz': sf * 1e3, 'chips': rows,
                    'median_before': float(np.median(Fb)), 'median_after': float(np.median(Fa)),
                    'frac_after_ge_099': float(np.mean(Fa >= 0.99))})
        log(f"    sigma_fab = {sf*1e3:4.0f} MHz: F without recalibration (median) {np.median(Fb):.4f} -> "
            f"after recalibrating the pulse only {np.median(Fa):.4f}; chips with F>=0.99: "
            f"{int(np.sum(Fa >= 0.99))}/{n_chips}")
    return out


STEP3_FILES = {'step2': 'v5_step3_', 'step2b': 'v5_step3b_', 'step3opt': 'v5_step3opt_analysis_'}


def run_step3(name, out_dir=None, log=print, do_recal=True, src='step2', do_abs_noise=False, do_eta=True, suffix=''):
    """
    A-posteriori analysis of a solution:
      src='step2'    -> v5_step2_<name>.json    -> v5_step3_<name>.json
      src='step2b'   -> v5_step2b_<name>.json   -> v5_step3b_<name>.json
      src='step3opt' -> v5_step3opt_<name>.json -> v5_step3opt_analysis_<name>.json
    suffix (e.g. '_fmin'): analyse v5_<src>_<name><suffix>.json and save the analysis with the same suffix.
    The physical limits (e.g. TPROT_MAX) are the ones stored in the solution JSON.
    """
    o = json.load(open(path_for(f'v5_{src}_{name}{suffix}.json', out_dir)))
    with phys_override(**o.get('PHYS', {})):
        log(f"\n================ STEP 3 ({src}) — {name} (F = {o['fidelity']:.5f}, n = {o['n_opt']}, "
            f"gate {o['n_opt'] * (o['T_opt'] + 6 * o['sigma_opt']):.0f} ns) ================")
        t0 = time.perf_counter()
        res = {'step': 3, 'source': src, 'transition': name, 'step2_fidelity': o['fidelity'],
               'n_opt': o['n_opt'], 'NOISE': NOISE, 'ETA': ETA, 'PHYS': dict(PHYS)}
        fc = res['floquet'] = floquet_contribution(name, o)
        log(f"  Floquet contribution: F with pulse {fc['F_drive']:.4f}; without pulse max "
            f"{fc['F_static_gate']:.4f} within the gate time ({PHYS['TPROT_MAX']:.0f} ns), "
            f"{fc['F_static_max']:.4f} within {fc['t_window_ns']:.0f} ns (at t = {fc['t_static_ns']:.0f} ns) "
            f"-> C = {fc['floquet_contribution']:.4f}")
        log("  Decoherence:")
        res['lindblad'] = lindblad_analysis(name, o, log=log)
        if do_eta:
            log("  Relative quasi-static noise (eta scan, same model as the original paper):")
            res['eta_scan'] = eta_scan(name, o, log=log)
        if do_abs_noise:
            log("  Absolute quasi-static noise (fixed n = n_opt):")
            res['quasi_static'] = quasi_static_noise(name, o, log=log)
        if do_recal:
            log("  Recalibration test (fabrication spread on the qubits, only the pulse is re-optimised):")
            res['recalibration'] = recalibration_test(name, o, log=log)
        res['time_s'] = time.perf_counter() - t0
    _save_json(res, out_dir, f'{STEP3_FILES[src]}{name}{suffix}.json', log)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 8. STEP 3 — NOISE-AWARE OPTIMISATION (starting from a Step 2b solution)
#
#   score = [ <F(n)>_eta - ALPHA_VAR * Var_eta(F(n)) ] * exp(-t_gate / TAU_DEC)
#           - anharmonicity penalty - static penalty (Floquet must stay essential)
#
#   * <.>_eta: average over a FIXED bank of K realisations of the relative noise (f_i, g_i -> x(1 + eta Z)),
#     n = that of the nominal candidate (in the lab n is fixed). ALPHA_VAR as in the cost function
#     of the original noise scan (C = 1 - F_mean + alpha Var).
#   * exp(-t_gate/TAU_DEC): decoherence. TAU_DEC is calibrated with the full Lindblad model on the starting
#     solution (tau = -t_gate / ln(F_Lindblad / F_unitary)); on the 2b solutions it is ~18 us for all
#     three transitions, so an exponential model in the gate duration is accurate.
#   * The final validation uses an INDEPENDENT bank (eta_scan, 128 samples) and the full Lindblad model.
#   * Interruptible: checkpoints in checkpoints/3opt_<name>_*.pkl, resume by re-running the same cell.
# ══════════════════════════════════════════════════════════════════════════════
NOISE_OPT = dict(
    ETA_OPT=1e-4,          # relative noise in the objective (~0.5-0.65 MHz on the qubits: pessimistic scenario)
    K=12,                  # noise realisations per candidate (fixed bank, common random numbers)
    BANK_SEED=2024,        # seed of the optimisation bank (validation uses eta_scan, seed 0)
    ALPHA_VAR=0.1,         # weight of the variance
    USE_DEC=True,          # include decoherence in the objective
    TAU_DEC=None,          # ns; None = calibrated with Lindblad on the starting solution
    POPSIZE=16,
    SIGMA0=0.1,            # random/local restarts (normalised space)
    SIGMA0_WARM=0.01,      # LOCAL search around the starting solution: with 0.05 (~125 MHz) the 1100 transition
                           # immediately left its narrow resonance and CMA got lost
    MAX_TIME=14400.0,      # s, total time (summed over all sessions); 1100 ~1 min/generation at K=12
    MAX_EVALS=30000,
    BATCH=96,              # systems per GPU call
)


def make_objective_noise(name, tau_dec, ties=None, K=None, f_prescreen=None, f_min=None, w_fmin=10.0):
    """
    Noise-aware objective (see above). ties: resonance constraints (reduced space, e.g. mirror):
    the noise is still applied to EVERY site independently, so it breaks the symmetry as in the lab.
    K: realisations (default NOISE_OPT['K']). f_prescreen: candidates with nominal F < f_prescreen
    are not simulated with noise and get <F> = F/2 (global search: saves most of the time spent on
    useless candidates, and the threshold still pushes the search above it).
    f_min: soft constraint on the NOMINAL F, score -= w_fmin * max(0, f_min - F) (refinement run_step3_fmin).
    """
    qt = qutrit(name)
    ties = ties or {}
    sp = step2_space(ties)
    eta, K = NOISE_OPT['ETA_OPT'], (NOISE_OPT['K'] if K is None else K)
    rng = np.random.default_rng(NOISE_OPT['BANK_SEED'])
    Zf, Zg = rng.normal(size=(K, N_SITES)), rng.normal(size=(K, N_SITES - 1))

    def obj(X):
        f, g, A, s, T, pen = step2_physical(sp.decode(X), ties)
        B = len(f)
        c, _ = fidelity_curve(qt, f, g, A, s, T, prec=P32, unit=UNIT_PHYS, n_max=N_MAX_CYCLES)
        Fn, nb = best_over_n(c, N_MIN_CYCLES, n_max_allowed(T, s), tol=STEP2B['N_EARLY_TOL'])
        Fs = static_max_batch(qt, f, g)
        Fm, Fv = Fn / 2, np.zeros(B)
        idx = np.arange(B) if f_prescreen is None else np.flatnonzero(Fn >= f_prescreen)
        if len(idx):
            b = len(idx)
            rep = lambda a: np.repeat(np.asarray(a)[idx], K)
            cp = fidelity_curve_batched(qt, (f[idx, None, :] * (1 + eta * Zf[None])).reshape(b * K, N_SITES),
                                        (g[idx, None, :] * (1 + eta * Zg[None])).reshape(b * K, N_SITES - 1),
                                        rep(A), rep(s), rep(T), batch=NOISE_OPT['BATCH'], prec=P32,
                                        unit=UNIT_PHYS, n_max=int(nb[idx].max()))
            Fk = cp[np.arange(b * K), np.repeat(nb[idx], K) - 1].reshape(b, K)
            Fm, Fv = Fm.copy(), Fv.copy()
            Fm[idx], Fv[idx] = Fk.mean(axis=1), Fk.var(axis=1)
        tg = nb * (T + 6 * s)
        dec = np.exp(-tg / tau_dec) if NOISE_OPT['USE_DEC'] else np.ones(B)
        score = (Fm - NOISE_OPT['ALPHA_VAR'] * Fv) * dec - pen - static_penalty(Fs)
        if f_min is not None:
            score = score - w_fmin * np.maximum(0.0, f_min - Fn)
        return -score, {'F': Fn, 'Fmean': Fm, 'Fstd': np.sqrt(Fv), 'Fdec': Fm * dec, 'Fstat': Fs,
                        'n': nb, 'gate': tg}
    return obj


def calibrate_tau_dec(name, o, log=print):
    """Effective decoherence time from solution o: F_L = F_U exp(-t_gate / tau)."""
    oq = OpenQutrit(name)
    *_, n_opt = _sol(o)
    FL = oq.evolve(o, n_cycles=n_opt)['F'][-1]
    FU = oq.evolve(o, n_cycles=n_opt, T1=np.full(N_SITES, np.inf), Tphi=np.full(N_SITES, np.inf))['F'][-1]
    tg = n_opt * (o['T_opt'] + 6 * o['sigma_opt'])
    tau = -tg / math.log(FL / FU)
    log(f"Decoherence calibration: unitary F {FU:.5f}, Lindblad {FL:.5f}, gate {tg:.0f} ns "
        f"-> tau_eff = {tau/1e3:.1f} us")
    return tau


def run_step3_opt(name, src='step2b', out_dir=None, fresh=False, analyse=True, log=print):
    """
    Noise-aware optimisation starting from v5_<src>_<name>.json (default: the Step 2b solution).
    Interruptible: re-running with the same parameters resumes from the checkpoint (fresh=True: from scratch).
    Saves v5_step3opt_<name>.json (same keys as the other solutions) and, if analyse, the a-posteriori
    analysis v5_step3opt_analysis_<name>.json (Lindblad + eta scan with an independent bank).
    """
    if fresh:
        clear_checkpoints(out_dir, '3opt', name)
    o0 = json.load(open(path_for(f'v5_{src}_{name}.json', out_dir)))
    with phys_override(**o0.get('PHYS', {})):
        log(f"\n================ STEP 3 — noise-aware optimisation: {name} (start: {src}, "
            f"F = {o0['fidelity']:.5f}, gate {o0['protocol_time_ns']:.0f} ns) ================")
        log("NOISE_OPT: " + ", ".join(f"{k}={v}" for k, v in NOISE_OPT.items()))
        meta_fn = ckpt_path(out_dir, '3opt', name, 'meta').replace('.pkl', '.json')
        tau = NOISE_OPT['TAU_DEC']
        if tau is None:
            meta = json.load(open(meta_fn)) if os.path.exists(meta_fn) else {}
            if meta.get('source') == src and meta.get('NOISE') == json.loads(json.dumps(NOISE)):
                tau = meta['tau_dec_ns']
                log(f"tau_eff from checkpoint: {tau/1e3:.1f} us")
            else:
                tau = calibrate_tau_dec(name, o0, log)
                json.dump({'tau_dec_ns': tau, 'source': src, 'NOISE': NOISE}, open(meta_fn, 'w'))
        sp = step2_space({})
        obj = make_objective_noise(name, tau)
        x0 = step2_encode(o0)
        f0, e0 = obj(x0[None])
        start = {k: float(v[0]) for k, v in e0.items()}
        start['score'] = float(-f0[0])
        log("Start: " + ", ".join(f"{k}={v:.4g}" for k, v in start.items()))
        b = run_cma(obj, sp.dim, warm_starts=[x0], popsize=NOISE_OPT['POPSIZE'], sigma0=NOISE_OPT['SIGMA0'],
                    sigma0_warm=NOISE_OPT['SIGMA0_WARM'], max_evals=NOISE_OPT['MAX_EVALS'],
                    max_time=NOISE_OPT['MAX_TIME'], max_restarts=40, local_restarts=True, seed=7, log=log,
                    checkpoint=ckpt_path(out_dir, '3opt', name, 'cma'),
                    sig_extra=dict(src=src, tau=round(tau, 3), x0=np.round(x0, 10).tolist(),
                                   **{k: NOISE_OPT[k] for k in ('ETA_OPT', 'K', 'BANK_SEED', 'ALPHA_VAR', 'USE_DEC')},
                                   **{k: STEP2B[k] for k in ('F_STATIC_MAX', 'PEN_STATIC_W', 'STATIC_TMAX')}))
        f, g, A, s, T, pen = decode_one(sp, b['x'])
        R = final_evaluation(name, f, g, A, s, T, n_tol=STEP2B['N_EARLY_TOL'], log=log)
        fx = floquet_extra(name, f, g, R['F'])
        final = {k: float(v) for k, v in b['extra'].items()}
        final['score'] = float(-b['f'])
        log("End:   " + ", ".join(f"{k}={v:.4g}" for k, v in final.items()))
        log(f"Without pulse: max F {fx['F_static_long_window']:.4f} -> Floquet contribution "
            f"C = {fx['floquet_contribution']:.4f}")
        log_audit(R['audit'], log)
        out = solution_json(name, '3opt', f, g, A, s, T, R, dict(
            fx, penalty=pen, NOISE_OPT=dict(NOISE_OPT), tau_dec_ns=tau, start_from=f'v5_{src}_{name}.json',
            objective_start=start, objective_final=final, evals=b['evals'], opt_time_s=b['time'],
            STEP2B=dict(STEP2B)))
    _save_json(out, out_dir, f'v5_step3opt_{name}.json', log)
    clear_checkpoints(out_dir, '3opt', name)
    if os.path.exists(meta_fn):
        os.remove(meta_fn)
    if analyse:
        run_step3(name, out_dir=out_dir, src='step3opt', do_recal=False, log=log)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 8b. STEP 3 — GLOBAL NOISE-AWARE SEARCH (for transitions where the local search is not enough)
#
#   The local search (run_step3_opt) cannot leave a narrow resonance: on 1100_0011 (all 4 qubits
#   at ~6.49 GHz, g ~ 40-90 MHz, small J_eff) it found no better candidate in 4 h.
#   Robustness scales as delta_f / J_eff: a different GEOMETRY is needed (lower qubits, larger g),
#   i.e. a global search with the noise already in the objective.
#   Phase G: space with the resonance constraints (mirror for 1100: 11 parameters), small K, prescreening
#            (candidates with nominal F < F_PRESCREEN are not simulated with noise), warm starts from the
#            solutions of the OTHER transitions (different geometries) + random IPOP restarts.
#   Phase L: local refinement in the full space with the standard objective (K = NOISE_OPT['K']),
#            local restarts only.
#   The result replaces v5_step3opt_<name>.json only if its score is better (the old one is renamed
#   *_local.json); otherwise it is saved as v5_step3opt_<name>_global.json.
# ══════════════════════════════════════════════════════════════════════════════
NOISE_GLOBAL = dict(
    K=4,                   # realisations in phase G (phase L uses NOISE_OPT['K'])
    F_PRESCREEN=0.5,       # below this nominal F no noisy simulation
    POPSIZE=16,
    SIGMA0=0.25,           # random restarts of phase G
    SIGMA0_WARM=0.1,       # warm starts from OTHER transitions: a large step is needed
    SIGMA0_LOCAL=0.01,     # phase L
    TIME_G=6 * 3600.0,     # s
    TIME_L=2 * 3600.0,     # s
    EVALS_G=600000,        # with prescreening the useless candidates are cheap: TIME_G decides
    EVALS_L=20000,
    WARM_FROM=('v5_step2b_{name}.json', 'v5_step2b_1001_0110.json', 'v5_step3opt_1001_0110.json',
               'v5_step2b_1010_0101.json', 'v5_step3opt_1010_0101.json'),
)


def _project_to_space(x_full, space):
    """Vector of the full space -> constrained space (dependent parameters dropped, offsets = 0)."""
    P = {k: float(v[0]) for k, v in step2_space({}).decode(x_full).items()}
    return np.clip(space.encode({k: float(np.clip(P.get(k, 0.0), lo, hi))
                                 for (k, lo, hi, _) in space.entries}), 0, 1)


def run_step3_global(name, out_dir=None, fresh=False, analyse=True, use_ties=True, log=print):
    """Global noise-aware search (phase G) + local refinement (phase L). See section 8b."""
    if fresh:
        clear_checkpoints(out_dir, '3g', name)
    NG = NOISE_GLOBAL
    o0 = json.load(open(path_for(f'v5_step2b_{name}.json', out_dir)))
    with phys_override(**o0.get('PHYS', {})):
        log(f"\n================ STEP 3 GLOBAL — {name} (ETA_OPT={NOISE_OPT['ETA_OPT']}) ================")
        log("NOISE_GLOBAL: " + ", ".join(f"{k}={v}" for k, v in NG.items()))
        meta_fn = ckpt_path(out_dir, '3g', name, 'meta').replace('.pkl', '.json')
        tau = NOISE_OPT['TAU_DEC']
        if tau is None:
            meta = json.load(open(meta_fn)) if os.path.exists(meta_fn) else {}
            if meta.get('NOISE') == json.loads(json.dumps(NOISE)):
                tau = meta['tau_dec_ns']
                log(f"tau_eff from checkpoint: {tau/1e3:.1f} us")
            else:
                tau = calibrate_tau_dec(name, o0, log)
                json.dump({'tau_dec_ns': tau, 'NOISE': NOISE}, open(meta_fn, 'w'))
        ties, kind = step2_ties(name, use_ties)
        spG, spF = step2_space(ties), step2_space({})
        warm_fns = [w.format(name=name) for w in NG['WARM_FROM']]
        warm_fns = [w for w in warm_fns if os.path.exists(path_for(w, out_dir))]
        warm = [_project_to_space(step2_encode(json.load(open(path_for(w, out_dir)))), spG) for w in warm_fns]
        log(f"Phase G: constraints = {kind} ({spG.dim} dim), K = {NG['K']}, warm starts from {warm_fns}")
        objG = make_objective_noise(name, tau, ties=ties, K=NG['K'], f_prescreen=NG['F_PRESCREEN'])
        common = {k: NOISE_OPT[k] for k in ('ETA_OPT', 'BANK_SEED', 'ALPHA_VAR', 'USE_DEC')}
        common.update({k: STEP2B[k] for k in ('F_STATIC_MAX', 'PEN_STATIC_W', 'STATIC_TMAX')}, tau=round(tau, 3))
        bG = run_cma(objG, spG.dim, warm_starts=warm, popsize=NG['POPSIZE'], sigma0=NG['SIGMA0'],
                     sigma0_warm=NG['SIGMA0_WARM'], max_evals=NG['EVALS_G'], max_time=NG['TIME_G'],
                     max_restarts=60, local_restarts=True, seed=17, log=log,
                     checkpoint=ckpt_path(out_dir, '3g', name, 'G'),
                     sig_extra=dict(common, K=NG['K'], ties=kind, prescreen=NG['F_PRESCREEN'], warm=warm_fns))
        xL0 = _to_full(spG, spF, ties, bG['x'])
        log(f"Phase G finished: score {-bG['f']:.4f} ({bG['evals']} evaluations, {bG['time']/3600:.2f} h). "
            f"Phase L: local refinement, full space ({spF.dim} dim), K = {NOISE_OPT['K']}")
        objL = make_objective_noise(name, tau)
        f0, e0 = objL(step2_encode(o0)[None])                 # reference: the 2b solution
        start = {k: float(v[0]) for k, v in e0.items()}
        start['score'] = float(-f0[0])
        bL = run_cma(objL, spF.dim, warm_starts=[xL0], popsize=NG['POPSIZE'], sigma0=NG['SIGMA0'],
                     sigma0_warm=NG['SIGMA0_LOCAL'], max_evals=NG['EVALS_L'], max_time=NG['TIME_L'],
                     max_restarts=40, random_restarts=False, seed=18, log=log,
                     checkpoint=ckpt_path(out_dir, '3g', name, 'L'),
                     sig_extra=dict(common, K=NOISE_OPT['K'], x0=np.round(xL0, 10).tolist()))
        f, g, A, s, T, pen = decode_one(spF, bL['x'])
        R = final_evaluation(name, f, g, A, s, T, n_tol=STEP2B['N_EARLY_TOL'], log=log)
        fx = floquet_extra(name, f, g, R['F'])
        final = {k: float(v) for k, v in bL['extra'].items()}
        final['score'] = float(-bL['f'])
        log("End:   " + ", ".join(f"{k}={v:.4g}" for k, v in final.items()))
        log(f"Without pulse: max F {fx['F_static_long_window']:.4f} -> Floquet contribution "
            f"C = {fx['floquet_contribution']:.4f}")
        log_audit(R['audit'], log)
        out = solution_json(name, '3opt', f, g, A, s, T, R, dict(
            fx, penalty=pen, NOISE_OPT=dict(NOISE_OPT), NOISE_GLOBAL=dict(NG), tau_dec_ns=tau,
            start_from='global search: ' + ', '.join(warm_fns), objective_start=start, objective_final=final,
            score_phase_G=float(-bG['f']), evals=bG['evals'] + bL['evals'], opt_time_s=bG['time'] + bL['time'],
            STEP2B=dict(STEP2B), search='global'))
    # comparison with the current 3opt solution (same tau: calibrated on the 2b solution)
    fn_cur = path_for(f'v5_step3opt_{name}.json', out_dir)
    old = json.load(open(fn_cur)) if os.path.exists(fn_cur) else None
    s_old = old['objective_final']['score'] if old and 'objective_final' in old else -np.inf
    if final['score'] > s_old:
        log(f"Global score {final['score']:.4f} > current {s_old:.4f}: it becomes the 3opt solution")
        if old is not None:
            os.replace(fn_cur, path_for(f'v5_step3opt_{name}_local.json', out_dir))
            fa = path_for(f'v5_step3opt_analysis_{name}.json', out_dir)
            if os.path.exists(fa):
                os.replace(fa, path_for(f'v5_step3opt_analysis_{name}_local.json', out_dir))
        _save_json(out, out_dir, f'v5_step3opt_{name}.json', log)
        if analyse:
            run_step3(name, out_dir=out_dir, src='step3opt', do_recal=False, log=log)
    else:
        log(f"Global score {final['score']:.4f} <= current {s_old:.4f}: the 3opt solution is kept; "
            f"the global result is saved separately")
        _save_json(out, out_dir, f'v5_step3opt_{name}_global.json', log)
    clear_checkpoints(out_dir, '3g', name)
    if os.path.exists(meta_fn):
        os.remove(meta_fn)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 8c. STEP 3 — LOCAL REFINEMENT WITH A CONSTRAINT ON THE NOMINAL F
#
#   Noise-optimised solutions can trade nominal F for robustness (1100 global search:
#   F = 0.984, <F>(1e-4) = 0.93). Here the run restarts from v5_step3opt_<name>.json with the SAME
#   objective (same tau, same bank, K = NOISE_OPT['K']) plus the soft constraint
#       score -= W * max(0, F_MIN - F_nominal)
#   with local restarts only. The result does NOT replace the 3opt solution: it goes to
#   v5_step3opt_<name>_fmin.json (+ analysis v5_step3opt_analysis_<name>_fmin.json); choose afterwards.
# ══════════════════════════════════════════════════════════════════════════════
NOISE_FMIN = dict(
    F_MIN=0.9905,          # threshold on the nominal F (0.99 + margin for the final float64 evaluation)
    W=10.0,                # 0.001 of F below threshold costs 0.01 of score
    POPSIZE=16,
    SIGMA0_LOCAL=0.01,     # restarts alternating 0.01 and 0.0033 (normalised space)
    MAX_TIME=2 * 3600.0,   # s
    MAX_EVALS=20000,
)


def run_step3_fmin(name, out_dir=None, fresh=False, analyse=True, log=print):
    """Local refinement of v5_step3opt_<name>.json with nominal F >= NOISE_FMIN['F_MIN'] (section 8c)."""
    if fresh:
        clear_checkpoints(out_dir, '3f', name)
    NF = NOISE_FMIN
    src_fn = f'v5_step3opt_{name}.json'
    o0 = json.load(open(path_for(src_fn, out_dir)))
    with phys_override(**o0.get('PHYS', {})):
        log(f"\n================ STEP 3 — refinement with nominal F >= {NF['F_MIN']}: {name} "
            f"(start {src_fn}: F = {o0['fidelity']:.5f}, gate {o0['protocol_time_ns']:.0f} ns) ================")
        log("NOISE_FMIN: " + ", ".join(f"{k}={v}" for k, v in NF.items()))
        tau = o0.get('tau_dec_ns') or NOISE_OPT['TAU_DEC'] or calibrate_tau_dec(name, o0, log)
        log(f"tau_eff = {tau/1e3:.1f} us (same as the starting solution)")
        sp = step2_space({})
        obj = make_objective_noise(name, tau, f_min=NF['F_MIN'], w_fmin=NF['W'])
        x0 = step2_encode(o0)
        f0, e0 = obj(x0[None])
        start = {k: float(v[0]) for k, v in e0.items()}
        start['score'] = float(-f0[0])
        log("Start: " + ", ".join(f"{k}={v:.4g}" for k, v in start.items()))
        b = run_cma(obj, sp.dim, warm_starts=[x0], popsize=NF['POPSIZE'], sigma0=NF['SIGMA0_LOCAL'],
                    sigma0_warm=NF['SIGMA0_LOCAL'], max_evals=NF['MAX_EVALS'], max_time=NF['MAX_TIME'],
                    max_restarts=40, random_restarts=False, seed=23, log=log,
                    checkpoint=ckpt_path(out_dir, '3f', name, 'L'),
                    sig_extra=dict(tau=round(tau, 3), x0=np.round(x0, 10).tolist(), F_MIN=NF['F_MIN'], W=NF['W'],
                                   **{k: NOISE_OPT[k] for k in ('ETA_OPT', 'K', 'BANK_SEED', 'ALPHA_VAR', 'USE_DEC')},
                                   **{k: STEP2B[k] for k in ('F_STATIC_MAX', 'PEN_STATIC_W', 'STATIC_TMAX')}))
        f, g, A, s, T, pen = decode_one(sp, b['x'])
        R = final_evaluation(name, f, g, A, s, T, n_tol=STEP2B['N_EARLY_TOL'], log=log)
        fx = floquet_extra(name, f, g, R['F'])
        final = {k: float(v) for k, v in b['extra'].items()}
        final['score'] = float(-b['f'])
        log("End:   " + ", ".join(f"{k}={v:.4g}" for k, v in final.items()))
        log(f"Without pulse: max F {fx['F_static_long_window']:.4f} -> Floquet contribution "
            f"C = {fx['floquet_contribution']:.4f}")
        log_audit(R['audit'], log)
        out = solution_json(name, '3opt', f, g, A, s, T, R, dict(
            fx, penalty=pen, NOISE_OPT=dict(NOISE_OPT), NOISE_FMIN=dict(NF), tau_dec_ns=tau, start_from=src_fn,
            objective_start=start, objective_final=final, evals=b['evals'], opt_time_s=b['time'],
            STEP2B=dict(STEP2B), search='local with F_MIN'))
    _save_json(out, out_dir, f'v5_step3opt_{name}_fmin.json', log)
    clear_checkpoints(out_dir, '3f', name)
    if analyse:
        run_step3(name, out_dir=out_dir, src='step3opt', do_recal=False, log=log, suffix='_fmin')
    return out
