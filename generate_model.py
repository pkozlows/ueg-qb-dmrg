"""Write the quasiboson model inputs for the 14e19o UEG cell (rs=4, 19 plane waves).

    python generate_model.py              # -> models/model_14e19o_k0.npz

H_el uses the probe-charge-Ewald HF eigenvalues `hfes_prew` (occupied levels carry -|v_M|,
the Koopmans/charged-excitation convention of McClain Fig 1a). The plasmons are per-q dRPA@HF
bosons (Omega_{nu q}, g_nu(q)) from `drpa_chol` on the same `hfes_prew` (see README "Madelung").

Provenance only: needs the group's private pyqcpbc / ueg / pyscf_gw stack (path below).
Everything downstream (run_tddmrg.py) needs only the npz.
Cleaned from gw-ac/renormalize.py (same SCF, same build_modes).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UEG_DIR = os.environ.get("QCPBC_UEG_DIR", "/n/home03/pkozlowski/qcpbc/libpbc/gw_tests/ueg")

RS = 4.0
N_ELEC = 14
NPW = N_ELEC + 5                # 19 plane waves

# dummy pyqcpbc input: only provides the SCF driver; the UEG integrals are injected below
QCPBC_INP = """$lattice
3
0.0 3.8370888985118827 3.8370888985118827
3.8370888985118827 0.0 3.8370888985118827
3.8370888985118827 3.8370888985118827 0.0
$end
$unitcell
ABSOLUTE
0 1
H 0.0 0.0 0.0
H 0.0 0.0 0.74
$end
$rem
jobtype = sp
method = HF
kecut = 1
scf_convergence = 8
scf_algorithm = gdm
pseudo = GTH-LDA
basis = SZV-GTH
input_bohr = true
$end
"""


def build_modes(mochol, moes, nocc, ueg, drpa_chol, tol=1e-12):
    """Per-q dRPA bosons as qb_dmrg.build_model modes: {"nu", "q", "Omega", "g"}.
    g_{q,mu} = |V_{k,k-q}^mu(q)| is k-independent for plane-wave orbitals."""
    modes = []
    for iq, q in enumerate(ueg.qvecs):
        pair = next(((p, m) for p, k in enumerate(ueg.basis)
                     if (m := ueg.lookup_basis(k - q)) is not None), None)
        if pair is None:
            continue
        exc_q, V_q = drpa_chol(mochol[[iq]], moes, nocc)
        g = np.abs(V_q[pair[0], pair[1], :])
        keep = g**2 > tol          # drop ph pairs outside the q sector: Omega = bare gap, g = 0
        qt = tuple(int(x) for x in q)
        for nu, (Om, gm) in enumerate(zip(exc_q[keep].real, g[keep])):
            modes.append(dict(nu=nu, q=qt, Omega=float(Om), g=float(gm)))
    return modes


def run_hf(SCF, minimizer, prog, hcore, cholJ, cholK, nbsf, nocc):
    """RHF on the UEG integrals, shifted Ewald + probe charge (occupied levels carry -|v_M|)."""
    class FockBuilder(SCF.fock_builder):
        def make(self, dm):
            dma = dm[0]
            J = 2.0 * np.einsum("prX,qsX,qs->pr", cholJ, cholJ, dma, optimize=True)
            K = -1.0 * np.einsum("prX,qsX,rs->pq", cholK, cholK, dma, optimize=True)
            h = hcore
            e1 = 2.0 * np.trace(h @ dma)
            eJ, eK = np.trace(J @ dma), np.trace(K @ dma)
            self.m_fock = [h + J + K]
            self.m_ecore = e1.real
            self.m_ej = [eJ.real]
            self.m_ek = [eK.real]
            return (e1 + eJ + eK).real

    dm0 = np.random.random((nbsf, nbsf))
    dm0 += dm0.T
    mf = SCF.scf(prog, FockBuilder(prog), nbsf, nocc, nocc, np.eye(nbsf), np.eye(nbsf))
    mf.set_dm([dm0])
    solver = minimizer()
    solver.cfg("algorithm", "diis")
    solver.cfg("maxiter", 1000)
    solver.cfg("tol", 1e-9)
    solver.run(mf)
    mf.extra_diagon_fock()
    return np.array(np.squeeze(mf.get_eorb()[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", default=os.path.join(HERE, "models", "model_14e19o_k0.npz"))
    ap.add_argument("--seed", type=int, default=0, help="random SCF guess")
    args = ap.parse_args()
    save = args.save

    import pyqcpbc
    from pyqcpbc.OPT import minimizer
    from pyqcpbc import SCF
    sys.path.append(UEG_DIR)
    from ueg import UEG, npw_to_kecut
    from pyscf_gw import drpa_chol
    from qb_dmrg import save_inputs
    np.random.seed(args.seed)

    prog = pyqcpbc.prog(QCPBC_INP, verbose=True)
    nocc = N_ELEC // 2
    kecut, kfac = npw_to_kecut(npw=NPW, rs=RS, n_elec=N_ELEC)
    opts = dict(nup=nocc, ndown=nocc, rs=RS, ecut=kecut / kfac**2, thermal=True)
    ueg_inc = UEG(opts, verbose=0); ueg_inc.build(g0include=True)
    ueg_exc = UEG(opts, verbose=0); ueg_exc.build(g0include=False)
    nbsf = ueg_inc.nbasis
    xi = ueg_inc.xi()

    aochol_exc = (2 * ueg_exc.chol_vecs.toarray()).reshape(nbsf, nbsf, -1).transpose((2, 0, 1)).copy()
    assert np.max(np.abs(np.imag(aochol_exc))) < 1e-10
    cholJ = (2 * ueg_inc.chol_vecsJ.toarray()).reshape(nbsf, nbsf, -1)
    cholK = (2 * ueg_inc.chol_vecsK.toarray()).reshape(nbsf, nbsf, -1)
    hcore = ueg_inc.H1[0]

    hfes_prew = run_hf(SCF, minimizer, prog, hcore, cholJ, cholK, nbsf, nocc)
    modes = build_modes(aochol_exc, hfes_prew, nocc, ueg_exc, drpa_chol)   # dRPA@HF
    kpts = [tuple(int(x) for x in k) for k in ueg_inc.basis]
    eps_hf = {k: e for k, e in zip(kpts, hfes_prew)}
    Om = np.array([m["Omega"] for m in modes]) * 27.2114
    g = np.array([m["g"] for m in modes])
    print(f"dRPA@HF: {len(modes)} modes, Omega {Om.min():.3f}..{Om.max():.3f} eV, "
          f"|g| {g.min():.3e}..{g.max():.3e} Ha; HF 2*sum(eps_occ) = {2*np.sum(hfes_prew[:nocc]):.10f} Ha")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    save_inputs(save, kpts, eps_hf, modes, ueg_inc.nelec, bath="drpa@hf", rs=RS,
                hfes_prew=hfes_prew, madelung_xi=xi)
    print(f"wrote {save}")


if __name__ == "__main__":
    main()
