"""
floquet.three_qubit — 3-site system of the original paper (Table 1): q1 - c - q2, transfer |1,0,0> -> |0,0,1>
(one excitation from qubit 1 to qubit 2 through the bus/coupler), reformulated with the Step 2b/3 physics:

    H = 2*pi*[ sum_i ( f_i n_i + alpha/2 n_i(n_i-1) ) + sum_i g_i (a_i+a_i^+)(a_{i+1}+a_{i+1}^+)
               - 2 A e^{-t^2/2 s^2} n_c ]                                   (GHz, ns; three-level transmons)

    * coupler ABOVE the qubits (dispersive), g <= 150 MHz, sigma >= 2 ns, excursion 2A <= 2 GHz
      and no coupler-qubit crossing during the kick (same limits as floquet.chain.PHYS);
    * gate <= 450 ns; penalty if the transfer also happens WITHOUT the pulse (Floquet essential);
    * objective with relative noise eta = 1e-4 and decoherence, as in the 7-site Step 3.

Reuses the functions of floquet.chain WITHOUT modifying them (Yoshida kick, F(n) curves, CMA with checkpoints):
the 3-site chain is a class "compatible" with the Transition objects of the 7-site module.
Everything is small (D <= 27): use_cpu() moves the computation to the CPU and leaves the GPU free.

Typical use (notebook 'notebooks/three_qubit_system.ipynb'):
    from floquet import three_qubit as s4
    s4.use_cpu()
    out = s4.run_step4()            # search + checks + Lindblad + eta scan + comparison with Table 1
"""
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from . import chain as fv

# ══════════════════════════════════════════════════════════════════════════════
# 0. Chain q1 - c - q2
# ══════════════════════════════════════════════════════════════════════════════
NS = 3
Q4 = [0, 2]
C4 = 1
NAME = '10_01'
SRC, TGT = (1, 0, 0), (0, 0, 1)

# Table 1 of the original paper (units of the original code: H = -w/2 sz + g sx sx, no 2*pi, delta kick)
ORIGINAL = dict(w=(4.4, 4.5, 4.6),            # (q1, bus, q2): in the original code the order was (q1, q2, bus)
                g=(0.4, 0.401), lam=4.0, T=10.945250619215415, n=447, F_paper=0.9999894054811106)

PHYS4 = dict(fv.PHYS)
PHYS4.update(TPROT_MAX=450.0)
STATIC4 = dict(TMAX=1500.0, DT=0.25,
               F_MAX=0.1,     # tolerated F without pulse. Stricter than for the 7-site chain (0.3): with 0.3 a test
                              # run found F_static = 0.28 (C = 0.72); for the 3-qubit example the drive must be essential
               W=1.0)
N_MIN, N_MAX = 3, 30
TOL_N = 5e-4                  # first n with F >= F_max - TOL_N (shorter gate)

OPT4 = dict(
    ETA_OPT=1e-4, BANK_SEED=2024, ALPHA_VAR=0.1,
    TAU_DEC=20e3,             # ns: effective decoherence in the objective (checked a posteriori with Lindblad)
    K_G=8, PRESCREEN=0.5,     # phase G (global): noise realisations, prescreening threshold
    K_L=24,                   # phase L (local)
    POPSIZE=16, SIGMA0=0.25, SIGMA0_LOCAL=0.01,
    TIME_G=1800.0, TIME_L=900.0, EVALS_G=400000, EVALS_L=40000,
)
OUT_RES, OUT_AN, OUT_FIG = 'results/step4', 'analysis/step4', 'figures/step4'


def use_cpu():
    """Compute on the CPU (the matrices are small): the GPU stays free for other runs."""
    fv.DEVICE = torch.device('cpu')
    fv.GPU = False


def path4(fn, kind='res', base=None):
    """Path of a Step 4 file: kind = 'res' (solutions), 'an' (analyses), 'fig' (figures)."""
    base = fv.DATA_ROOT if base is None else base
    d = os.path.join(base, {'res': OUT_RES, 'an': OUT_AN, 'fig': OUT_FIG}[kind])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fn)


class Chain3:
    """
    3-site chain compatible with the Transition objects of floquet.chain (diag, xx_flat, v, torch_consts, ...).
    levels=3: transmon (alpha); levels=2: qubit. nmax: truncation on N_tot. both_parities: for Lindblad.
    """

    def __init__(self, levels=3, nmax=4, both_parities=False, alpha=None):
        alpha = PHYS4['ALPHA'] if alpha is None else alpha
        par = sum(SRC) % 2
        states = [s for s in itertools.product(range(levels), repeat=NS)
                  if sum(s) <= nmax and (both_parities or sum(s) % 2 == par)]
        pos = {s: j for j, s in enumerate(states)}
        S = np.array(states)
        self.states, self.D, self.levels = states, len(states), levels
        self.src_i, self.tgt_i = pos[SRC], pos[TGT]
        self.n = S.T.astype(np.float64)                                   # (3, D)
        self.anh = (alpha / 2.0 * self.n * (self.n - 1)).sum(axis=0) if levels > 2 else np.zeros(self.D)
        self.v = -2.0 * self.n[C4]
        self.leak_mask = (S == 2).any(axis=1)
        XX = np.zeros((NS - 1, self.D, self.D))
        for j, s in enumerate(states):
            for i in range(NS - 1):
                for di in (-1, 1):
                    ki = s[i] + di
                    if not 0 <= ki < levels:
                        continue
                    for dj in (-1, 1):
                        kj = s[i + 1] + dj
                        if not 0 <= kj < levels:
                            continue
                        t = list(s); t[i] = ki; t[i + 1] = kj; t = tuple(t)
                        if t in pos:
                            XX[i, pos[t], j] = math.sqrt(max(s[i], ki)) * math.sqrt(max(s[i + 1], kj))
        self.xx_flat = XX.reshape(NS - 1, -1)
        self._torch_cache = {}

    def diag(self, f):
        return f @ self.n + self.anh[None, :]

    def torch_consts(self):
        if not self._torch_cache:
            v = torch.tensor(self.v, dtype=torch.float64, device=fv.DEVICE)
            src = torch.zeros(self.D, dtype=torch.complex128, device=fv.DEVICE)
            src[self.src_i] = 1.0
            self._torch_cache = (v, src)
        return self._torch_cache


_TR = {}


def chain(levels=3, nmax=4, both_parities=False):
    key = (levels, nmax, both_parities, str(fv.DEVICE))
    if key not in _TR:
        _TR[key] = Chain3(levels, nmax, both_parities)
    return _TR[key]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Physical parametrisation (same constraints as the 7-site Step 2)
# ══════════════════════════════════════════════════════════════════════════════
def space4():
    P = PHYS4
    return fv.Space([('q1', *P['FQ'], False), ('q2', *P['FQ'], False), ('D', 0.0, 1.0, False),
                     ('g1', *P['G'], False), ('g2', *P['G'], False),
                     ('sigma', *P['SIGMA'], False), ('u', *P['U'], True), ('T', *P['T'], False)])


def excursion4(f):
    f = np.atleast_2d(f)
    gap = f[:, C4] - np.maximum(f[:, 0], f[:, 2])
    e = np.minimum(PHYS4['DF_MAX'], f[:, C4] - PHYS4['F_FLOOR'])
    return np.maximum(np.minimum(e, gap - PHYS4['KICK_DQC_MIN']), 0.0)


def physical4(P):
    """Parameters -> f (B,3), g (B,2), A, sigma, T, anharmonicity penalty."""
    B = len(P['sigma'])
    f = np.zeros((B, NS))
    f[:, 0], f[:, 2] = P['q1'], P['q2']
    m = np.maximum(f[:, 0], f[:, 2])
    dmax = np.minimum(PHYS4['DQC'][1], PHYS4['FC_MAX'] - m)
    f[:, C4] = m + PHYS4['DQC'][0] + np.clip(P['D'], 0, 1) * (dmax - PHYS4['DQC'][0])
    g = np.stack([P['g1'], P['g2']], axis=1)
    A = P['u'] * excursion4(f) / 2.0
    x = np.abs(np.abs(f[:, 0] - f[:, 2]) - abs(PHYS4['ALPHA']))
    pen = PHYS4['PEN_W'] * np.clip(1.0 - x / PHYS4['ALPHA_WIN'], 0, None) ** 2
    return f, g, A, np.asarray(P['sigma']), np.asarray(P['T']), pen


def encode4(o):
    f, g = np.asarray(o['omegas_opt']), np.asarray(o['couplings'])
    m = max(f[0], f[2])
    dmax = min(PHYS4['DQC'][1], PHYS4['FC_MAX'] - m)
    P = dict(q1=f[0], q2=f[2], D=(f[1] - m - PHYS4['DQC'][0]) / max(dmax - PHYS4['DQC'][0], 1e-12),
             g1=g[0], g2=g[1], sigma=o['sigma_opt'], T=o['T_opt'],
             u=2 * o['A_opt'] / max(float(excursion4(f)[0]), 1e-12))
    sp = space4()
    return np.clip(sp.encode({k: float(np.clip(P[k], lo, hi)) for (k, lo, hi, _) in sp.entries}), 0, 1)


def nmax_ok(T, s):
    return np.clip(np.floor(PHYS4['TPROT_MAX'] / (np.asarray(T) + 6 * np.asarray(s))), 0, N_MAX).astype(int)


def curves(tr, f, g, A, s, T, prec=fv.P64, n_max=N_MAX, n_tr=None):
    f = np.atleast_2d(np.asarray(f, float))
    g = np.atleast_2d(np.asarray(g, float))
    if len(g) == 1 and len(f) > 1:
        g = np.repeat(g, len(f), axis=0)
    return fv.fidelity_curve(tr, f, g, np.atleast_1d(A), np.atleast_1d(s), np.atleast_1d(T), prec=prec,
                             unit=fv.UNIT_PHYS, n_max=n_max, n_tr=n_tr)[0]


def static4(tr, f, g, t_max=None, dt=None, return_time=False):
    """max_t F without pulse (A = 0)."""
    f = np.atleast_2d(np.asarray(f, float))
    g = np.atleast_2d(np.asarray(g, float))
    return fv.static_max_batch(tr, f, g, t_max=STATIC4['TMAX'] if t_max is None else t_max,
                               dt=STATIC4['DT'] if dt is None else dt, return_time=return_time)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Noise-aware objective and search (global phase G + local phase L)
# ══════════════════════════════════════════════════════════════════════════════
def make_objective4(K, prescreen=None):
    tr, sp = chain(), space4()
    eta = OPT4['ETA_OPT']
    rng = np.random.default_rng(OPT4['BANK_SEED'])
    Zf, Zg = rng.normal(size=(K, NS)), rng.normal(size=(K, NS - 1))

    def obj(X):
        f, g, A, s, T, pen = physical4(sp.decode(X))
        B = len(f)
        c = curves(tr, f, g, A, s, T)
        Fn, nb = fv.best_over_n(c, N_MIN, nmax_ok(T, s), tol=TOL_N)
        nb = np.maximum(nb, 1)
        Fs = static4(tr, f, g)
        Fm, Fv = Fn / 2, np.zeros(B)
        idx = np.arange(B) if prescreen is None else np.flatnonzero(Fn >= prescreen)
        if len(idx):
            b = len(idx)
            rep = lambda a: np.repeat(np.asarray(a)[idx], K)
            cp = curves(tr, (f[idx, None, :] * (1 + eta * Zf[None])).reshape(b * K, NS),
                        (g[idx, None, :] * (1 + eta * Zg[None])).reshape(b * K, NS - 1),
                        rep(A), rep(s), rep(T), n_max=int(nb[idx].max()))
            Fk = cp[np.arange(b * K), np.repeat(nb[idx], K) - 1].reshape(b, K)
            Fm, Fv = Fm.copy(), Fv.copy()
            Fm[idx], Fv[idx] = Fk.mean(axis=1), Fk.var(axis=1)
        tg = nb * (T + 6 * s)
        dec = np.exp(-tg / OPT4['TAU_DEC'])
        score = (Fm - OPT4['ALPHA_VAR'] * Fv) * dec - pen - STATIC4['W'] * np.clip(Fs - STATIC4['F_MAX'], 0, None)
        return -score, {'F': Fn, 'Fmean': Fm, 'Fstd': np.sqrt(Fv), 'Fstat': Fs, 'n': nb, 'gate': tg}
    return obj


def search4(out_dir=None, fresh=False, warm=(), log=print):
    """Phase G (global, random IPOP restarts, prescreening) + phase L (local, larger K). Interruptible."""
    if fresh:
        fv.clear_checkpoints(out_dir, '4', NAME)
    sp = space4()
    sig = dict(OPT4={k: OPT4[k] for k in ('ETA_OPT', 'BANK_SEED', 'ALPHA_VAR', 'TAU_DEC')},
               STATIC4=dict(STATIC4), PHYS4={k: (list(v) if isinstance(v, tuple) else v) for k, v in PHYS4.items()})
    log(f"Phase G: global search ({sp.dim} parameters), K = {OPT4['K_G']}, prescreening F >= {OPT4['PRESCREEN']}")
    bG = fv.run_cma(make_objective4(OPT4['K_G'], OPT4['PRESCREEN']), sp.dim, warm_starts=list(warm),
                    popsize=OPT4['POPSIZE'], sigma0=OPT4['SIGMA0'], sigma0_warm=0.1, max_evals=OPT4['EVALS_G'],
                    max_time=OPT4['TIME_G'], max_restarts=60, seed=41, log=log,
                    checkpoint=fv.ckpt_path(out_dir, '4', NAME, 'G'), sig_extra=dict(sig, K=OPT4['K_G'],
                                                                                    prescreen=OPT4['PRESCREEN']))
    log(f"Phase G finished: score {-bG['f']:.5f}. Phase L: local refinement, K = {OPT4['K_L']}")
    bL = fv.run_cma(make_objective4(OPT4['K_L']), sp.dim, warm_starts=[bG['x']], popsize=OPT4['POPSIZE'],
                    sigma0=OPT4['SIGMA0'], sigma0_warm=OPT4['SIGMA0_LOCAL'], max_evals=OPT4['EVALS_L'],
                    max_time=OPT4['TIME_L'], max_restarts=20, random_restarts=False, seed=42, log=log,
                    checkpoint=fv.ckpt_path(out_dir, '4', NAME, 'L'),
                    sig_extra=dict(sig, K=OPT4['K_L'], x0=np.round(bG['x'], 10).tolist()))
    return bG, bL


# ══════════════════════════════════════════════════════════════════════════════
# 3. Checks: qubit vs transmon, Trotter, truncation, audit, Floquet contribution
# ══════════════════════════════════════════════════════════════════════════════
def audit4(f, g, A, s, T, n):
    f, g = np.asarray(f, float), np.asarray(g, float)
    H1 = np.diag(f) + np.diag(g, 1) + np.diag(g, -1)
    _, V = np.linalg.eigh(H1)
    part = [float(V[C4, k] ** 2) for k in range(NS) if (V[Q4, k] ** 2).sum() > 0.5]
    dqc = [f[C4] - f[0], f[C4] - f[2]]
    d12 = abs(f[0] - f[2])
    Jeff = g[0] * g[1] / 2 * (1 / dqc[0] + 1 / dqc[1])
    return {'f_qubit_GHz': [float(f[0]), float(f[2])], 'f_coupler_GHz': float(f[C4]), 'g_GHz': g.tolist(),
            'Delta_q1q2_GHz': float(d12), 'dist_from_alpha_resonance_GHz': float(abs(d12 - abs(PHYS4['ALPHA']))),
            'min_Delta_qc_GHz': float(min(dqc)), 'max_g_over_Delta_qc': float(max(g[0] / dqc[0], g[1] / dqc[1])),
            'J_eff_dispersive_MHz': float(Jeff * 1e3),
            'max_coupler_participation_qubit_modes': float(max(part)),
            'coupler_excursion_2A_GHz': float(2 * A), 'min_coupler_freq_during_kick_GHz': float(f[C4] - 2 * A),
            'min_Delta_qc_during_kick_GHz': float(min(dqc) - 2 * A), 'sigma_ns': float(s),
            'pulse_bandwidth_GHz': float(1 / (2 * math.pi * s)), 'kick_window_ns': float(6 * s),
            'period_ns': float(T + 6 * s), 'protocol_time_ns': float(n * (T + 6 * s))}


def evaluate4(f, g, A, s, T, log=print):
    f, g = np.asarray(f, float), np.asarray(g, float)
    L = fv.n_trotter(A, s, fv.UNIT_PHYS, 2)
    cq3 = curves(chain(3, 4), f, g, A, s, T, n_tr=L)[0]
    cq3b = curves(chain(3, 4), f, g, A, s, T, n_tr=2 * L)[0]
    cq3n6 = curves(chain(3, 6), f, g, A, s, T, n_tr=L)[0]
    cqb = curves(chain(2, 3), f, g, A, s, T, n_tr=L)[0]
    Fb, nb = fv.best_over_n(cq3[None], N_MIN, nmax_ok(T, s), tol=TOL_N)
    n = int(nb[0])
    st_gate = static4(chain(), f, g, t_max=PHYS4['TPROT_MAX'], dt=0.05)
    st_long, t_long = static4(chain(), f, g, t_max=STATIC4['TMAX'], dt=0.05, return_time=True)
    UF = fv.floquet_operator(chain(3, 4), f[None], g[None], A, s, T, fv.UNIT_PHYS, n_tr=L)
    psi = np.zeros(UF.shape[0], complex); psi[chain(3, 4).src_i] = 1
    leak = []
    for _ in range(N_MAX):
        psi = UF @ psi
        leak.append(float(np.sum(np.abs(psi[chain(3, 4).leak_mask]) ** 2)))
    R = {'n': n, 'F': float(cq3[n - 1]), 'F_qubit': float(cqb[n - 1]), 'F_N6': float(cq3n6[n - 1]),
         'trotter_err': float(abs(cq3b[n - 1] - cq3[n - 1])), 'leak': leak[n - 1],
         'F_static_gate_window': float(st_gate[0]), 'F_static_long_window': float(st_long[0]),
         't_static_long_ns': float(t_long[0]), 'floquet_contribution': float(cq3[n - 1] - st_long[0]),
         'curve_qutrit': cq3.tolist(), 'curve_qubit': cqb.tolist(), 'leak_curve': leak,
         'audit': audit4(f, g, A, s, T, n)}
    log(f"Check: transmon F={R['F']:.6f} at n={n} (N<=6: {R['F_N6']:.6f}, leakage {R['leak']:.1e}, "
        f"Trotter error {R['trotter_err']:.1e}); qubit model {R['F_qubit']:.6f}")
    log(f"Without pulse: max F {R['F_static_gate_window']:.4f} within {PHYS4['TPROT_MAX']:.0f} ns, "
        f"{R['F_static_long_window']:.4f} within {STATIC4['TMAX']:.0f} ns -> Floquet contribution "
        f"C = {R['floquet_contribution']:.4f}")
    return R


# ══════════════════════════════════════════════════════════════════════════════
# 4. Decoherence (Lindblad) and relative quasi-static noise — same conventions as the 7-site chain
# ══════════════════════════════════════════════════════════════════════════════
class Open3:
    """drho/dt = -i[H,rho] + sum_i D[sqrt(1/T1_i) a_i] + D[sqrt(2/Tphi_i) n_i]; exact channels + Strang."""

    def __init__(self, nmax=4):
        self.tr = chain(3, nmax, both_parities=True)
        st = self.tr.states
        pos = {s: j for j, s in enumerate(st)}
        self.D, self.n = self.tr.D, self.tr.n
        self.lower = {}
        for i in range(NS):
            for k in (1, 2):
                src = np.array([j for j, s in enumerate(st) if s[i] >= k])
                dst = np.array([pos[st[j][:i] + (st[j][i] - k,) + st[j][i + 1:]] for j in src])
                self.lower[(i, k)] = (src, dst)

    def rates(self, scale=1.0):
        N = fv.NOISE
        T1 = np.array([N['T1_C'] if k == C4 else N['T1_Q'] for k in range(NS)]) * scale
        Tp = np.array([N['TPHI_C'] if k == C4 else N['TPHI_Q'] for k in range(NS)]) * scale
        return T1, Tp

    def dissipate(self, rho, tau, T1, Tphi):
        if tau <= 0:
            return rho
        M = np.zeros((self.D, self.D))
        for i in range(NS):
            if np.isfinite(Tphi[i]):
                M += (self.n[i][:, None] - self.n[i][None, :]) ** 2 / Tphi[i]
        rho = rho * np.exp(-tau * M)
        for i in range(NS):
            if not np.isfinite(T1[i]):
                continue
            p = 1.0 - math.exp(-tau / T1[i])
            ni = self.n[i]
            k0 = (1.0 - p) ** (ni / 2.0)
            new = rho * np.outer(k0, k0)
            for k in (1, 2):
                src, dst = self.lower[(i, k)]
                nn = ni[src].astype(int)
                w = np.sqrt(fv._COMB[nn, k] * p ** k * (1.0 - p) ** (nn - k))
                new[np.ix_(dst, dst)] += rho[np.ix_(src, src)] * np.outer(w, w)
            rho = new
        return rho

    def evolve(self, o, n_cycles, T1=None, Tphi=None, seg_max=None):
        f, g, A, s, T = (np.asarray(o['omegas_opt'], float), np.asarray(o['couplings'], float),
                         o['A_opt'], o['sigma_opt'], o['T_opt'])
        seg_max = fv.NOISE['SEG_MAX'] if seg_max is None else seg_max
        if T1 is None:
            T1, Tphi = self.rates()
        H = fv.build_H0(self.tr, f[None], g[None], fv.UNIT_PHYS)
        ev, evc = fv.diagonalize(H)
        Uk = fv.kick_unitary(self.tr, ev, evc, fv._t64(A), fv._t64(s), fv.n_trotter(A, s, fv.UNIT_PHYS, 2),
                             fv.P64, fv.UNIT_PHYS)[0].cpu().numpy()
        E, V = ev[0].cpu().numpy(), evc[0].cpu().numpy()
        n_seg = max(1, int(math.ceil(T / seg_max)))
        tau = T / n_seg
        Us = (V * np.exp(-1j * E * tau)) @ V.T
        tk = 6 * s
        rho = np.zeros((self.D, self.D), complex)
        rho[self.tr.src_i, self.tr.src_i] = 1
        F, trerr = [], 0.0
        for _ in range(n_cycles):
            rho = self.dissipate(rho, tk / 2, T1, Tphi)
            rho = Uk @ rho @ Uk.conj().T
            rho = self.dissipate(rho, (tk + tau) / 2, T1, Tphi)
            for m in range(n_seg):
                rho = Us @ rho @ Us.conj().T
                rho = self.dissipate(rho, tau if m < n_seg - 1 else tau / 2, T1, Tphi)
            F.append(float(np.real(rho[self.tr.tgt_i, self.tr.tgt_i])))
            trerr = max(trerr, abs(np.trace(rho).real - 1))
        return {'F': F, 'trace_err': trerr,
                'min_eig_final': float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min())}


def lindblad4(o, log=print):
    oq, n = Open3(), o['n_opt']
    inf = np.full(NS, np.inf)
    U = oq.evolve(o, n, T1=inf, Tphi=inf)
    R = oq.evolve(o, n)
    R2 = oq.evolve(o, n, seg_max=fv.NOISE['SEG_MAX'] / 2)
    scan = []
    for sc in fv.NOISE['T1_SCALES']:
        T1, Tp = oq.rates(sc)
        scan.append({'scale': sc, 'T1_qubit_us': fv.NOISE['T1_Q'] * sc / 1e3,
                     'F': oq.evolve(o, n, T1=T1, Tphi=Tp)['F'][-1]})
    out = {'D_open': oq.D, 'F_unitary_check': U['F'][-1], 'F_lindblad': R['F'][-1],
           'splitting_err': abs(R2['F'][-1] - R['F'][-1]), 'trace_err': R['trace_err'],
           'min_eig_final': R['min_eig_final'], 'T1_scan': scan,
           'rates_ns': {k: fv.NOISE[k] for k in ('T1_Q', 'TPHI_Q', 'T1_C', 'TPHI_C')}}
    log(f"Lindblad (D={oq.D}): unitary F {out['F_unitary_check']:.5f} -> with decoherence {out['F_lindblad']:.5f} "
        f"(splitting error {out['splitting_err']:.1e}, |tr-1| {out['trace_err']:.1e})")
    return out


def eta_scan4(o, eta_list=None, n_samples=None, seed=0, log=print):
    eta_list = fv.ETA['ETA_LIST'] if eta_list is None else eta_list
    n_samples = fv.ETA['N_SAMPLES'] if n_samples is None else n_samples
    f0, g0 = np.asarray(o['omegas_opt'], float), np.asarray(o['couplings'], float)
    A, s, T, n = o['A_opt'], o['sigma_opt'], o['T_opt'], o['n_opt']
    rng = np.random.default_rng(seed)
    Zf, Zg = rng.normal(size=(n_samples, NS)), rng.normal(size=(n_samples, NS - 1))
    res = []
    for eta in eta_list:
        C = curves(chain(), f0[None] * (1 + eta * Zf), g0[None] * (1 + eta * Zg), np.full(n_samples, A),
                   np.full(n_samples, s), np.full(n_samples, T))
        Fx = C[:, n - 1]
        Fmx, _ = fv.best_over_n(C, N_MIN, np.full(n_samples, max(int(nmax_ok(T, s)), N_MIN)))   # as in the 7-site chain
        q = np.sort(Fx)
        res.append({'eta': eta, 'df_qubit_MHz': eta * float(np.mean(f0[Q4])) * 1e3, 'mean': float(Fx.mean()),
                    'std': float(Fx.std()), 'p20': float(np.percentile(Fx, 20)),
                    'cvar20': float(q[:max(1, n_samples // 5)].mean()), 'min': float(Fx.min()),
                    'mean_maxn': float(Fmx.mean()), 'std_maxn': float(Fmx.std()),
                    'curve_mean': C.mean(0).tolist(), 'curve_std': C.std(0).tolist()})
        log(f"  eta={eta:7.0e} (df_q ~ {res[-1]['df_qubit_MHz']:6.3f} MHz): <F>={res[-1]['mean']:.4f} "
            f"std={res[-1]['std']:.4f} p20={res[-1]['p20']:.4f} min={res[-1]['min']:.4f}")
    return {'levels': res, 'n_opt': n, 'n_samples': n_samples}


# ══════════════════════════════════════════════════════════════════════════════
# 5. The original Table 1 in the same language (comparison for the referee)
# ══════════════════════════════════════════════════════════════════════════════
def original_table1(n_max=None, log=print):
    """
    Reproduces Table 1 (qubit model, H = -w/2 sz + g sx sx, delta kick exp(-i lam sz_bus), no 2*pi)
    and re-evaluates it in the three-level transmon model with alpha = -0.2 in the SAME units (reading:
    'GHz with time in units of 1/(2 pi) ns'). Also reports the physical quantities criticised by the referee.
    """
    o = ORIGINAL
    w, g = np.array(o['w']), np.array(o['g'])
    n_max = o['n'] + 20 if n_max is None else n_max
    out = {}
    for lev, lab in ((2, 'qubit'), (3, 'qutrit')):
        tr = Chain3(lev, nmax=4, alpha=PHYS4['ALPHA'])
        H = (g[None] @ tr.xx_flat).reshape(tr.D, tr.D) + np.diag(tr.diag(w[None])[0])
        E, V = np.linalg.eigh(H)
        Uf = (V * np.exp(-1j * E * o['T'])) @ V.conj().T
        # sz_bus = 1 - 2 n_bus  ->  exp(-i lam sz) = e^{-i lam} exp(+2 i lam n_bus)
        Uk = np.diag(np.exp(2j * o['lam'] * tr.n[C4]))
        UF = Uf @ Uk
        psi = np.zeros(tr.D, complex); psi[tr.src_i] = 1
        Fc = []
        for _ in range(n_max):
            psi = UF @ psi
            Fc.append(float(abs(psi[tr.tgt_i]) ** 2))
        out[lab] = {'F_at_n_paper': Fc[o['n'] - 1], 'F_max': float(max(Fc)), 'n_at_max': int(np.argmax(Fc) + 1)}
    dqc = (float(w[1] - w[0]), float(w[1] - w[2]))
    out['issues'] = {'bus_between_qubits': bool(min(w[0], w[2]) < w[1] < max(w[0], w[2])),
                     'Delta_qc': [float(x) for x in dqc], 'g_over_Delta_qc': float(max(g) / min(abs(np.array(dqc)))),
                     'Delta_q1q2_vs_alpha': [float(abs(w[0] - w[2])), abs(PHYS4['ALPHA'])],
                     'delta_kick': True, 'n_cycles': o['n'],
                     'gate_time_if_GHz_ns': o['n'] * o['T'] / (2 * math.pi),
                     'gate_time_if_rad_per_ns': o['n'] * o['T']}
    out['paper'] = dict(o)
    log(f"Original Table 1: qubit model F(n={o['n']}) = {out['qubit']['F_at_n_paper']:.6f} "
        f"(paper {o['F_paper']:.6f}); transmon model (alpha=-0.2) F(n={o['n']}) = "
        f"{out['qutrit']['F_at_n_paper']:.4f}, max {out['qutrit']['F_max']:.4f} at n={out['qutrit']['n_at_max']}")
    log(f"  bus between the qubits: {out['issues']['bus_between_qubits']}, Delta_qc = {np.round(dqc, 3).tolist()}, "
        f"g/Delta = {out['issues']['g_over_Delta_qc']:.1f}, {o['n']} cycles")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. Full pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_step4(out_dir=None, fresh=False, warm_from=None, log=print):
    """Search (interruptible) + checks + Lindblad + eta scan + Table 1. Saves the JSONs in results/ and analysis/step4."""
    t0 = time.perf_counter()
    log(f"\n================ STEP 4 — 3 sites q1-c-q2, |100> -> |001>, gate <= {PHYS4['TPROT_MAX']:.0f} ns ================")
    log("OPT4: " + ", ".join(f"{k}={v}" for k, v in OPT4.items()))
    warm = []
    if warm_from:
        warm = [encode4(json.load(open(p))) for p in warm_from if os.path.exists(p)]
    bG, bL = search4(out_dir, fresh, warm, log)
    sp = space4()
    f, g, A, s, T, pen = (x[0] for x in physical4(sp.decode(bL['x'])))
    R = evaluate4(f, g, A, s, T, log)
    o = {'step': 4, 'transition': NAME, 'sites': 'q1 - c - q2',
         'units': 'GHz and ns: H = 2*pi*[ f_i n_i + alpha/2 n_i(n_i-1) + g_i (a+a^+)(a+a^+) - 2 A e^{-t^2/2s^2} n_c ]',
         'omegas_opt': f.tolist(), 'couplings': g.tolist(), 'A_opt': float(A), 'sigma_opt': float(s),
         'T_opt': float(T), 'n_opt': R['n'], 'fidelity': R['F'], 'fidelity_qubit_at_n_opt': R['F_qubit'],
         'fidelity_qutrit_N6': R['F_N6'], 'leakage_qutrit_N4': R['leak'], 'trotter_err_qutrit': R['trotter_err'],
         'F_static_gate_window': R['F_static_gate_window'], 'F_static_long_window': R['F_static_long_window'],
         'floquet_contribution': R['floquet_contribution'], 'protocol_time_ns': R['audit']['protocol_time_ns'],
         'audit': R['audit'], 'penalty': float(pen), 'curve_qutrit': R['curve_qutrit'], 'curve_qubit': R['curve_qubit'],
         'objective_final': {k: float(v) for k, v in bL['extra'].items()}, 'score': float(-bL['f']),
         'score_phase_G': float(-bG['f']), 'evals': bG['evals'] + bL['evals'], 'opt_time_s': bG['time'] + bL['time'],
         'OPT4': dict(OPT4), 'PHYS4': {k: (list(v) if isinstance(v, tuple) else v) for k, v in PHYS4.items()},
         'STATIC4': dict(STATIC4)}
    log("Physics audit:")
    for k, v in R['audit'].items():
        log(f"   {k:40s} = {np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)}")
    with open(path4(f'v5_step4_{NAME}.json', 'res', out_dir), 'w') as fh:
        json.dump(o, fh, indent=1)
    log("A-posteriori analysis:")
    an = {'lindblad': lindblad4(o, log), 'eta_scan': eta_scan4(o, log=log), 'original_table1': original_table1(log=log),
          'time_s': time.perf_counter() - t0}
    with open(path4(f'v5_step4_analysis_{NAME}.json', 'an', out_dir), 'w') as fh:
        json.dump(an, fh, indent=1)
    fv.clear_checkpoints(out_dir, '4', NAME)
    log(f"Saved {path4(f'v5_step4_{NAME}.json', 'res', out_dir)} and the analysis ({(time.perf_counter()-t0)/60:.1f} min)")
    return o, an
