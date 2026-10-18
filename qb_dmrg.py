"""Quasiboson (bosonized UEG) model in Renormalizer: Hamiltonian, DMRG ground state,
hole spectral function by real-time TD-DMRG, and dense exact-diagonalization helpers
for checking both on tiny instances.

Model (built by build_model):
    H = sum_{k,s} eps_k n_{ks} + sum_{q,nu} Omega_{q,nu} b+_{q,nu} b_{q,nu}
        + sum_{q,nu} g_{q,nu} sum_{k,s} c+_{k+q,s} c_{k,s} (b_{q,nu} + b+_{-q,nu})
Fermions are Jordan-Wigner mapped onto BasisHalfSpin sites; bosons are BasisSHO
sites truncated at nbas_boson levels (n_max = nbas_boson - 1). No SCF happens here;
generate_model.py builds the inputs and save_inputs() writes them.
"""
import json
import os
import time

import numpy as np

if not hasattr(np, "product"):      # Renormalizer 0.0.11 uses np.product (removed in NumPy 2)
    np.product = np.prod
from renormalizer import Model, Op, Mps, Mpo, BasisSHO, BasisHalfSpin
from renormalizer.mps.gs import optimize_mps
from renormalizer.utils import EvolveConfig, EvolveMethod, CompressConfig, CompressCriteria, Quantity
np.seterr(all="warn")   # renormalizer sets divide/over/invalid="raise" globally at import


# ---------------------------------------------------------------------------
# Model inputs (written by generate_model.py, read by every DMRG stage)
# ---------------------------------------------------------------------------
def save_inputs(fname, kpts, eps_hf, modes, nelec, **extra):
    kpts = [tuple(int(x) for x in k) for k in kpts]
    np.savez(fname,
             kpts=np.array(kpts, dtype=int),
             eps=np.array([eps_hf[k] for k in kpts], dtype=float),
             mode_nu=np.array([m["nu"] for m in modes], dtype=int),
             mode_q=np.array([m["q"] for m in modes], dtype=int).reshape(-1, 3),
             mode_Omega=np.array([m["Omega"] for m in modes], dtype=float),
             mode_g=np.array([m["g"] for m in modes], dtype=float),
             nelec=np.array(nelec, dtype=int),
             **extra)


def load_inputs(fname):
    d = np.load(fname)
    kpts = [tuple(int(x) for x in k) for k in d["kpts"]]
    eps_hf = {k: float(e) for k, e in zip(kpts, d["eps"])}
    modes = [dict(nu=int(nu), q=tuple(int(x) for x in q), Omega=float(Om), g=float(g))
             for nu, q, Om, g in zip(d["mode_nu"], d["mode_q"], d["mode_Omega"], d["mode_g"])]
    return kpts, eps_hf, modes, tuple(int(x) for x in d["nelec"])


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------
def hop(p, q, factor=1.0):
    """JW-mapped a+_p a_q on electron sites p, q (integers = JW order)."""
    if p == q:
        # n_p = sigma_- sigma_+ on the same site (projector onto |1>)
        return Op("sigma_- sigma_+", [p, p], factor, qn=[1, -1])
    lo, hi = min(p, q), max(p, q)
    sites = list(range(lo, hi + 1))
    symbols, qns = [], []
    for r in sites:
        if r == p:
            symbols.append("sigma_-"); qns.append(1)
        elif r == q:
            symbols.append("sigma_+"); qns.append(-1)
        else:
            symbols.append("Z"); qns.append(0)
    return Op(" ".join(symbols), sites, factor, qn=qns)


def attach_boson(el_op, symbol, bdof):
    """el_op (x) boson op. Built directly rather than with Op.__mul__, which calls
    np.product and breaks on NumPy >= 2."""
    return Op(el_op.symbol + " " + symbol, list(el_op.dofs) + [bdof],
              el_op.factor, qn=list(el_op.qn_list) + [0])


def build_model(kpts, eps_hf, modes, nbas_boson=6, interleave=True):
    """
    kpts   : list of k-points, each a tuple of ints (grid indices), e.g. (nx, ny, nz)
    eps_hf : dict  k -> HF orbital energy
    modes  : list of dicts {"nu": nu, "q": q(tuple), "Omega": float, "g": float}
             must contain q and -q for every q (for each nu)
    Returns (model, info) where info maps names to dofs.
    Fermion sites are ordered by eps_hf (k=0 first, shells contiguous, up/down adjacent).
    """
    kpts = sorted((tuple(k) for k in kpts), key=lambda k: (eps_hf[k], k))
    kset = set(kpts)
    spins = (0, 1)

    # electron spin-orbitals in JW order; dof name = integer JW index
    orb_index = {}
    for k in kpts:
        for s in spins:
            orb_index[(k, s)] = len(orb_index)

    mode_dof = {(m["nu"], tuple(m["q"])): ("pl", m["nu"], tuple(m["q"])) for m in modes}

    # --- basis (MPS site order). Order affects bond dimension, not correctness.
    el_basis = [BasisHalfSpin(i, sigmaqn=[0, 1]) for i in range(len(orb_index))]
    bo_basis = [BasisSHO(mode_dof[(m["nu"], tuple(m["q"]))], m["Omega"], nbas_boson) for m in modes]
    if interleave:
        basis, nb = [], len(bo_basis)
        stride = max(1, len(el_basis) // max(nb, 1))
        bi = 0
        for i, b in enumerate(el_basis):
            basis.append(b)
            if (i + 1) % stride == 0 and bi < nb:
                basis.append(bo_basis[bi]); bi += 1
        basis += bo_basis[bi:]
    else:
        basis = el_basis + bo_basis

    terms = []
    # H_el
    for k in kpts:
        for s in spins:
            p = orb_index[(k, s)]
            terms.append(hop(p, p, eps_hf[k]))
    # H_pl
    for m in modes:
        terms.append(Op(r"b^\dagger b", mode_dof[(m["nu"], tuple(m["q"]))], m["Omega"]))
    # H_ep
    for m in modes:
        nu, q, g = m["nu"], tuple(m["q"]), m["g"]
        if g == 0:
            continue
        mq = tuple(-x for x in q)
        if (nu, mq) not in mode_dof:
            raise ValueError(f"mode (nu={nu}, -q={mq}) missing; needed for b+_(nu,-q)")
        for k in kpts:
            kq = tuple(a + b for a, b in zip(k, q))
            if kq not in kset:          # k+q outside the finite basis -> drop
                continue
            for s in spins:
                el = hop(orb_index[(kq, s)], orb_index[(k, s)], g)
                terms.append(attach_boson(el, r"b^\dagger", mode_dof[(nu, mq)]))
                terms.append(attach_boson(el, "b", mode_dof[(nu, q)]))

    return Model(basis, terms), dict(orb_index=orb_index, mode_dof=mode_dof)


# ---------------------------------------------------------------------------
# Ground state
# ---------------------------------------------------------------------------
def dmrg_ground_state(model, nelec, M=160, procedure=None, nroots=1, m_init=20,
                      e_rtol=1e-6, e_atol=1e-8, mps0=None, mpo=None, n_sweeps=12):
    """M: final bond dimension. procedure overrides the default schedule if given.
    mps0: warm start (same model) -- one light-noise sweep, then noiseless sweeps.
    Returns (energies per sweep, mps, converged) where converged mirrors Renormalizer's
    own sweep test (two lowest sweep energies within e_rtol/e_atol)."""
    mpo = mpo if mpo is not None else Mpo(model)
    if mps0 is not None:
        mps = mps0.copy()
        if procedure is None:
            procedure = [[M, 0.1]] + [[M, 0]] * n_sweeps
    else:
        mps = Mps.random(model, nelec, m_init)
        if procedure is None:
            # ramp up with noise, then noiseless sweeps at the target M
            procedure = [[max(M // 4, 1), 0.4], [max(M // 2, 1), 0.2],
                         [max(3 * M // 4, 1), 0.1]] + [[M, 0]] * n_sweeps
    if len(procedure) < 2:      # Renormalizer only records the result MPS from sweep 2 on
        procedure = list(procedure) + [procedure[-1]]
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    mps.optimize_config.nroots = nroots
    mps.optimize_config.e_rtol = e_rtol
    mps.optimize_config.e_atol = e_atol
    energies, mps = optimize_mps(mps, mpo)
    converged = False
    for n in range(2, len(energies) + 1):
        if procedure[n - 1][1] == 0:
            v1, v2 = sorted(energies[:n])[:2]
            if np.allclose(v1, v2, rtol=e_rtol, atol=e_atol):
                converged = True
                break
    return energies, mps, converged


def hf_mps(model, info, kpts, eps_hf, nocc):
    """HF determinant (x) plasmon vacuum as a bond-dimension-1 MPS: both spins of the nocc
    lowest-eps_hf k-points occupied. Its energy in H_qb is exactly 2*sum(eps_occ)."""
    occ = sorted(kpts, key=lambda k: eps_hf[k])[:nocc]
    cond = {info["orb_index"][(k, s)]: 1 for k in occ for s in (0, 1)}
    return Mps.hartree_product_state(model, cond)


def seeded_hf_mps(model, info, kpts, eps_hf, nocc, mpo, m_seed=16, coef=1e-3):
    """DMRG starting point: HF (x) vacuum plus a small admixture of H|HF> (bond dim m_seed).
    Plain HF is a fixed point of 2-site DMRG here: every coupling a+_{k+q} a_k b touches three
    sites and the noise cannot open new particle-number sectors on the bonds, so DMRG stays at
    E_HF exactly. A random start stalls above E_HF instead (seen at nbas=2 up to M=1024)."""
    mps = hf_mps(model, info, kpts, eps_hf, nocc)
    mps.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=m_seed)
    return mps.expand_bond_dimension(mpo, coef=coef, include_ex=False)


def gs_energy(model, mps, mpo=None):
    """<gs|H|gs> of the returned MPS (rather than the lowest sweep energy)."""
    mpo = mpo if mpo is not None else Mpo(model)
    return float(np.real(mps.expectation(mpo)))


def number_mpo(model, info):
    return Mpo(model, [hop(p, p) for p in info["orb_index"].values()])


def gs_observables(model, info, mps, nbas_boson, k0=(0, 0, 0)):
    out = dict(N=float(np.real(mps.expectation(number_mpo(model, info)))),
               n_k0=float(np.real(mps.expectation(Mpo(model, hop(info["orb_index"][(k0, 0)],
                                                                    info["orb_index"][(k0, 0)]))))))
    # occupation of the highest SHO level (n = nbas_boson - 1), summed over modes, from
    # one-site RDMs (BasisSHO.op_mat has no projector / n^k symbols)
    bsites = [i for i, b in enumerate(model.basis) if isinstance(b, BasisSHO)]
    rdm = mps.calc_1site_rdm(bsites) if bsites else {}
    out["top_level_occ"] = float(sum(np.real(rdm[i][nbas_boson - 1, nbas_boson - 1])
                                     for i in bsites)) if nbas_boson > 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# Hole (photoemission) spectral function by real-time evolution
#   C(t)   = <0| c_p^+ e^{-i(H-E0)t} c_p |0> = sum_n w_n e^{+i omega_n t},
#            w_n = |<n|c_p|0>|^2, omega_n = E0 - E_n^{N-1}  (absolute removal energy)
#   A_p(w) = (1/pi) Re int_0^T dt e^{+i w t} conj(C(t)) e^{-eta t}
#          = sum_n w_n (eta/pi) / ((w - omega_n)^2 + eta^2)   as T -> infinity
# eta is the Lorentzian HWHM (the "-i eta" in Eq. 2 of McClain et al.).
# ---------------------------------------------------------------------------
def annihilator(p):
    """JW-mapped a_p on electron site p."""
    return Op(" ".join(["Z"] * p + ["sigma_+"]), list(range(p + 1)), qn=[0] * p + [-1])


def _td_configure(phi, method, m_max):
    phi.evolve_config = EvolveConfig(getattr(EvolveMethod, method))
    phi.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=m_max)
    return phi


def _is_real_mps(mps):
    return all(np.isrealobj(np.asarray(mt.array)) for mt in mps) and np.isreal(mps.coeff)


def hole_correlation_td(model, gs_mps, e0, p, dt=0.5, t_max=700.0, m_max=128,
                        method="auto", two_way=True, h_mpo=None,
                        ckpt=None, ckpt_every=50, log=print):
    """Return (t, C(t), weight) for removing an electron from spin-orbital p.

    method  : 'auto' (default) runs 2-site TDVP ('tdvp_ps2') while the bond dimension is
              below m_max, then switches to 1-site TDVP ('tdvp_ps') at fixed m_max; this
              avoids the random padding 1-site TDVP needs to grow from a small bond
              dimension. 'tdvp_ps' or 'tdvp_ps2' force one integrator throughout.
    two_way : H, |0> and c_p are real, so <0|c+ e^{-iHt} c|0> = phi(t/2)^T phi(t/2) with
              phi(tau) = e^{-iH tau} c|0> (bilinear, no complex conjugation). One evolution
              to t_max/2 then gives C on [0, t_max] with spacing 2*dt, and the MPS only has
              to carry the entanglement of time t/2. two_way=False is the textbook
              <phi(0)|phi(t)> (spacing dt, evolution to t_max).
    ckpt    : path prefix; '<ckpt>_C.npz' and '<ckpt>_mps.npz' are written every
              ckpt_every steps (temp + rename) and read back to resume.
    """
    h_mpo = h_mpo if h_mpo is not None else Mpo(model, offset=Quantity(e0))
    stride = 2 if two_way else 1
    nsteps = int(np.ceil(t_max / (stride * dt) - 1e-9))

    phi = Mpo(model, annihilator(p)).apply(gs_mps.copy())
    phi.canonicalise()
    weight = float(phi.norm) ** 2 / float(gs_mps.norm) ** 2   # = <n_p>, total spectral weight
    phi = phi.normalize("mps_and_coeff")
    if two_way and not (_is_real_mps(gs_mps) and _is_real_mps(phi)):
        raise ValueError("two_way needs a real ground state and real c_p|0>; use two_way=False")
    phi0 = phi.copy()

    C, start = [1.0 + 0j], 0
    if ckpt is not None and os.path.exists(ckpt + "_C.npz") and os.path.exists(ckpt + "_mps.npz"):
        d = np.load(ckpt + "_C.npz")
        C, start = list(d["C"] / d["weight"]), int(d["step"])
        phi = Mps.load(model, ckpt + "_mps.npz")
        log(f"resumed from step {start} (t = {start * stride * dt:.2f})")

    def integrator(phi):
        if method != "auto":
            return method
        return "tdvp_ps" if max(phi.bond_dims) >= m_max else "tdvp_ps2"

    current = integrator(phi)
    _td_configure(phi, current, m_max)
    if start == 0 and current == "tdvp_ps":
        phi = phi.expand_bond_dimension(h_mpo, include_ex=False)   # forced 1-site from the start
    log(f"integrator {current} (bond dim {max(phi.bond_dims)}, m_max {m_max})")

    def overlap(phi):
        if two_way:
            return complex(phi.dot(phi) * phi.coeff ** 2)
        return complex(phi0.conj().dot(phi) * np.conj(phi0.coeff) * phi.coeff)

    def dump(step):
        tmp = ckpt + "_mps.tmp.npz"
        phi.dump(tmp)
        os.replace(tmp, ckpt + "_mps.npz")
        tmp = ckpt + "_C.tmp.npz"
        np.savez(tmp, C=weight * np.array(C), step=step, weight=weight, dt=dt, stride=stride)
        os.replace(tmp, ckpt + "_C.npz")

    t0 = time.time()
    for i in range(start + 1, nsteps + 1):
        phi = phi.evolve(h_mpo, dt)
        C.append(overlap(phi))
        nxt = integrator(phi)
        if nxt != current:
            log(f"step {i}: bond dim reached {max(phi.bond_dims)} -> switching to {nxt}")
            current = nxt
            _td_configure(phi, current, m_max)
        if ckpt is not None and (i % ckpt_every == 0 or i == nsteps):
            dump(i)
        if i % ckpt_every == 0 or i == nsteps:
            log(f"step {i}/{nsteps} t={i * stride * dt:.1f} |C|={abs(C[-1]):.4e} "
                f"M={max(phi.bond_dims)} {time.time() - t0:.0f}s")
    t = np.arange(len(C)) * stride * dt
    return t, weight * np.array(C), weight


def exact_correlation(poles, weights, t):
    """Lehmann C(t) = sum_n w_n e^{+i omega_n t} for checking the TD route."""
    return (weights[None, :] * np.exp(1j * np.outer(t, poles))).sum(axis=1)


def spectrum_from_correlation(t, C, omegas, eta):
    """A(w) = (1/pi) Re int_0^T e^{+i w t} conj(C(t)) e^{-eta t} dt (trapezoid).
    Truncation at T leaves a relative ringing of order e^{-eta T}."""
    dt = t[1] - t[0]
    f = C.conj() * np.exp(-eta * t)
    f[0] *= 0.5
    f[-1] *= 0.5
    return (np.exp(1j * np.outer(omegas, t)) @ f).real * dt / np.pi


def lorentzian_sum(poles, weights, omegas, eta):
    return (weights[None, :] * (eta / np.pi)
            / ((omegas[:, None] - poles[None, :]) ** 2 + eta ** 2)).sum(axis=1)


# ---------------------------------------------------------------------------
# Dense exact diagonalization (tiny models only; Mpo.todense caps at dim 20000)
# ---------------------------------------------------------------------------
def dense_hole_spectrum(model, info, nelec_tot, p, omegas, eta):
    """Exact E0 and A_p(omega) (Lehmann sum) from the dense Hamiltonian."""
    H = Mpo(model).todense()
    assert np.allclose(H, H.conj().T, atol=1e-12), "H not Hermitian"
    Nop = np.real(np.diag(number_mpo(model, info).todense()))
    c = Mpo(model, annihilator(p)).todense()
    sN = np.where(np.abs(Nop - nelec_tot) < 1e-8)[0]
    sM = np.where(np.abs(Nop - (nelec_tot - 1)) < 1e-8)[0]
    eN, vN = np.linalg.eigh(H[np.ix_(sN, sN)])
    eM, vM = np.linalg.eigh(H[np.ix_(sM, sM)])
    gs = vN[:, 0]
    amp = vM.conj().T @ (c[np.ix_(sM, sN)] @ gs)
    w = np.abs(amp) ** 2
    poles = eN[0] - eM
    A = lorentzian_sum(poles, w, omegas, eta)
    return eN[0], A, dict(poles=poles, weights=w, dim=H.shape[0])


def dumps_json(obj, fname):
    with open(fname + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(fname + ".tmp", fname)
