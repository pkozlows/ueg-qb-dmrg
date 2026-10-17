"""Hole spectral function A_{k=0}(omega) of the 14e19o quasiboson Hamiltonian by TD-DMRG.

HF band + dRPA@HF plasmons (models/model_14e19o_k0.npz). All 19 k-points, 14 electrons
and every plasmon mode are kept.

Stages (each reads the model npz, never re-runs SCF):
  check    tiny sub-model vs dense ED: DMRG E0, TD C(t) vs Lehmann C(t) (one- and two-way),
           A(w) vs Lorentzian sum; then a timing probe on the full model
                                                          -> check_report.json, timing.json
  gs       ground state: bond dimension M nested inside boson levels nbas
                                                          -> gs_scan.dat, gs_converged.json, gs_nb*_M*.npz
  td       one array task per point of the M and nbas ladders (see td_tasks)
                                                          -> td/<tag>_final.npz, td/<tag>.json
  analyze  A(w), peaks, convergence tables and figures    -> A_k0_td.dat, convergence.md, *.png
  ntasks   print the number of td array tasks
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HA2EV = 27.2114
ETA = 0.2 / HA2EV                         # Lorentzian HWHM (Ha), McClain Fig. 1a
K0 = (0, 0, 0)

# ---- ground-state ladders
NB_LIST = [1, 2, 3, 4, 5, 6]              # SHO levels per mode (n_max = nbas - 1)
M_LIST = [32, 64, 128, 256, 512, 1024]
TOL_M = 1e-6                              # Ha, E0 change between successive M
TOL_NB = 1e-6                             # Ha, E0 change between successive nbas
TOP_OCC_TOL = 1e-6                        # summed occupation of the highest SHO level

# ---- time evolution: fixed grid, reference point and ladders (one parameter at a time)
DT = 0.5                                  # a.u.; TDVP step
T_MAX = 700.0                             # a.u.; e^{-eta T} = 6e-3
TWO_WAY = True
METHOD = "auto"                           # 2-site TDVP until bond dim = M, then 1-site
M_REF = 256                               # TD bond dimension
NB_REF = 4                                # n_max = 3
M_TD = [64, 128, 256, 512]
NB_TD = [2, 3, 4, 5, 6]

OMEGAS_EV = np.arange(-20.0, 0.0 + 1e-9, 0.01)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def done(workdir, stage):
    with open(os.path.join(workdir, f"{stage}.done"), "w") as f:
        f.write(time.strftime("%c") + "\n")


def import_q():
    import qb_dmrg as Q
    logging.getLogger("renormalizer").setLevel(logging.WARNING)
    return Q


# ---------------------------------------------------------------------------
def stage_check(args):
    Q = import_q()
    from renormalizer import Mpo
    kpts, eps_hf, modes, nelec = Q.load_inputs(args.model)
    rep = dict()

    # --- tiny sub-model: k0 and +-x; modes with q = +-x (largest g first) within the
    #     Mpo.todense cap (2^6 fermion states x 3^nmodes <= 20000 -> <= 5 modes)
    xs = [(1, 0, 0), (-1, 0, 0)]
    sub_k = [K0] + xs
    sub_eps = {k: eps_hf[k] for k in sub_k}
    by_nu = {}
    for m in modes:
        if m["q"] in xs:
            by_nu.setdefault(m["nu"], []).append(m)
    pairs = sorted((v for v in by_nu.values() if len(v) == 2), key=lambda v: -v[0]["g"])
    sub_modes = [m for v in pairs[:2] for m in v]
    ne = 2                                     # k0 doubly occupied
    omegas = OMEGAS_EV / HA2EV
    for label, nbas in [("nb3", 3), ("nb1_HF", 1)]:
        model, info = Q.build_model(sub_k, sub_eps, sub_modes, nbas_boson=nbas)
        p = info["orb_index"][(K0, 0)]
        E0x, Ax, d = Q.dense_hole_spectrum(model, info, ne, p, omegas, ETA)
        en, mps, conv = Q.dmrg_ground_state(model, ne, M=64, e_rtol=0, e_atol=1e-12)
        E0 = Q.gs_energy(model, mps)
        r = dict(dim=d["dim"], E0_exact=E0x, E0_dmrg=E0, dE=E0 - E0x, dmrg_converged=conv,
                 n_modes=len(sub_modes), weight_exact=float(d["weights"].sum()))
        if nbas == 1:
            # boson vacuum: single HF pole at eps_k0 with unit weight
            ipk = int(np.argmax(d["weights"]))
            r["hf_pole_err"] = float(d["poles"][ipk] - eps_hf[K0])
            r["hf_weight"] = float(d["weights"][ipk])
            assert abs(r["hf_pole_err"]) < 1e-10 and abs(r["hf_weight"] - 1) < 1e-10, r
        # TD route: C(t) vs Lehmann at the production dt, both propagation schemes.
        # M=128 exceeds the exact bond dimension of this model, so any error is TDVP's.
        for two_way in (True, False):
            t, C, w = Q.hole_correlation_td(model, mps, E0, p, dt=DT, t_max=200.0, m_max=128,
                                            method=METHOD, two_way=two_way, log=lambda s: None)
            Cx = Q.exact_correlation(d["poles"], d["weights"], t)
            A_td = Q.spectrum_from_correlation(t, C, omegas, ETA)
            A_lehmann_T = Q.spectrum_from_correlation(t, Cx, omegas, ETA)
            tag = "two_way" if two_way else "one_way"
            r[f"{tag}_weight_err"] = float(w - d["weights"].sum())
            r[f"{tag}_max_dC"] = float(np.max(np.abs(C - Cx)))
            r[f"{tag}_max_dA_tdvp"] = float(np.max(np.abs(A_td - A_lehmann_T)) / HA2EV)
            r[f"{tag}_max_dA_truncT200"] = float(np.max(np.abs(A_lehmann_T - Ax)) / HA2EV)
        log(f"check {label}: {r}")
        assert abs(r["dE"]) < 1e-9, r
        assert r["two_way_max_dC"] < 1e-6 and r["one_way_max_dC"] < 1e-6, r
        rep[label] = r
    os.makedirs(args.workdir, exist_ok=True)
    Q.dumps_json(rep, os.path.join(args.workdir, "check_report.json"))

    # --- timing probe on the full model (nbas=3)
    model, info = Q.build_model(kpts, eps_hf, modes, nbas_boson=3)
    h = Mpo(model)
    ne_tot = sum(nelec)
    tm = dict(n_sites=len(model.basis), n_modes=len(modes), n_fermion_sites=2 * len(kpts))
    t0 = time.time()
    en, mps, _ = Q.dmrg_ground_state(model, ne_tot, M=64, mpo=h)
    tm["gs_M64_full_run_s"] = time.time() - t0
    tm["gs_M64_sweeps"] = len(en)
    for M in [128, 256]:
        t0 = time.time()
        en, mps, _ = Q.dmrg_ground_state(model, ne_tot, M=M, mpo=h, mps0=mps,
                                         procedure=[[M, 0.0], [M, 0.0]])
        tm[f"gs_sweep_M{M}_s"] = (time.time() - t0) / 2
    E0 = Q.gs_energy(model, mps, h)
    p = info["orb_index"][(K0, 0)]
    for M in [64, 128]:
        t0 = time.time()
        Q.hole_correlation_td(model, mps, E0, p, dt=DT, t_max=5 * 2 * DT, m_max=M,
                              method="tdvp_ps", two_way=True, log=lambda s: None)
        tm[f"td_step_M{M}_s"] = (time.time() - t0) / 5
    steps = int(np.ceil(T_MAX / (2 * DT)))
    tm["td_ref_estimate_h_at_M128"] = tm["td_step_M128_s"] * steps / 3600
    log(f"timing: {tm}")
    Q.dumps_json(tm, os.path.join(args.workdir, "timing.json"))
    done(args.workdir, "check")


# ---------------------------------------------------------------------------
def stage_gs(args):
    Q = import_q()
    from renormalizer import Mpo
    kpts, eps, modes, nelec = Q.load_inputs(args.model)
    ne = sum(nelec)
    hf_energy = 2 * sum(sorted(eps.values())[: nelec[0]])
    os.makedirs(args.workdir, exist_ok=True)
    scan = open(os.path.join(args.workdir, "gs_scan.dat"), "a")
    scan.write("# nb  M  E0(Ha)  dE_M  N  n_k0up  top_level_occ  sweeps  dmrg_conv  wall_s\n")
    per_nb = {}
    E_prev_nb, m_start, nb_conv = None, 0, None
    for nb in NB_LIST:
        model, info = Q.build_model(kpts, eps, modes, nbas_boson=nb)
        h = Mpo(model)
        E_prev, mps, M_conv = None, None, None
        ms = M_LIST[m_start:]
        for M in ms:
            t0 = time.time()
            en, mps, dconv = Q.dmrg_ground_state(model, ne, M=M, mpo=h, mps0=mps,
                                                 e_rtol=0, e_atol=TOL_M / 10)
            E = Q.gs_energy(model, mps, h)
            obs = Q.gs_observables(model, info, mps, nb)
            dE = np.nan if E_prev is None else E - E_prev
            scan.write(f"{nb} {M} {E:.12f} {dE:+.3e} {obs['N']:.8f} {obs['n_k0']:.8f} "
                       f"{obs['top_level_occ']:.3e} {len(en)} {int(dconv)} {time.time()-t0:.0f}\n")
            scan.flush()
            log(f"nb={nb} M={M} E0={E:.10f} dE={dE:+.2e} {obs} dmrg_conv={dconv}")
            if nb == 1:
                assert abs(E - hf_energy) < 1e-8, (E, hf_energy)
            fname = os.path.join(args.workdir, f"gs_nb{nb}_M{M}.npz")
            mps.dump(fname)
            if E_prev is not None and abs(dE) < TOL_M and dconv:
                M_conv = M
                break
            E_prev = E
        if M_conv is None:
            log(f"WARNING: M not converged at nb={nb} up to M={ms[-1]}")
            M_conv = ms[-1]
        per_nb[nb] = dict(M=M_conv, E0=E, file=fname, obs=obs,
                          M_converged=bool(abs(dE) < TOL_M))
        m_start = max(M_LIST.index(M_conv) - 1, 0)   # next nb needs at least this M
        dE_nb = np.nan if E_prev_nb is None else E - E_prev_nb
        log(f"nb={nb}: M*={M_conv} E0={E:.10f} dE_nb={dE_nb:+.2e} top_occ={obs['top_level_occ']:.2e}")
        if (E_prev_nb is not None and abs(dE_nb) < TOL_NB
                and obs["top_level_occ"] < TOP_OCC_TOL and per_nb[nb]["M_converged"]):
            nb_conv = nb
            break
        E_prev_nb = E
    nb_converged = nb_conv is not None
    if not nb_converged:
        log(f"WARNING: nbas_boson not converged up to {NB_LIST[-1]}")
        nb_conv = max(per_nb)
    out = dict(nb=nb_conv, M=per_nb[nb_conv]["M"], E0=per_nb[nb_conv]["E0"],
               file=per_nb[nb_conv]["file"], nb_converged=nb_converged,
               per_nb={str(k): v for k, v in per_nb.items()}, hf_energy=hf_energy,
               tol_M=TOL_M, tol_nb=TOL_NB)
    Q.dumps_json(out, os.path.join(args.workdir, "gs_converged.json"))
    done(args.workdir, "gs")


# ---------------------------------------------------------------------------
def td_tasks():
    """The reference point, then one-parameter-at-a-time ladders in M and nbas."""
    ref = dict(M=M_REF, nb=NB_REF)
    tasks = [dict(ref, tag="ref", ladder="ref")]
    tasks += [dict(ref, M=M, tag=f"M{M}", ladder="M") for M in M_TD if M != M_REF]
    tasks += [dict(ref, nb=nb, tag=f"nb{nb}", ladder="nb") for nb in NB_TD if nb != NB_REF]
    return tasks


def td_ground_state(Q, args, model, h, nb, ne):
    """Reuse the gs-stage MPS when that nb was scanned; otherwise run DMRG at the
    converged gs bond dimension."""
    from renormalizer import Mps
    gs = load_json(os.path.join(args.workdir, "gs_converged.json"))
    g = gs["per_nb"].get(str(nb))
    if g is not None:
        return Mps.load(model, g["file"]), g["M"]
    M = gs["M"]
    en, mps, conv = Q.dmrg_ground_state(model, ne, M=M, mpo=h, e_rtol=0, e_atol=TOL_M / 10)
    if not conv:
        log(f"WARNING: GS for nb={nb} not converged at M={M}")
    return mps, M


def stage_td(args):
    Q = import_q()
    from renormalizer import Mpo
    task = td_tasks()[args.task]
    kpts, eps, modes, nelec = Q.load_inputs(args.model)
    ne = sum(nelec)
    tddir = os.path.join(args.workdir, "td")
    os.makedirs(tddir, exist_ok=True)
    model, info = Q.build_model(kpts, eps, modes, nbas_boson=task["nb"])
    h = Mpo(model)
    mps, M_gs = td_ground_state(Q, args, model, h, task["nb"], ne)
    E0 = Q.gs_energy(model, mps, h)
    p = info["orb_index"][(K0, 0)]
    ckpt = os.path.join(tddir, task["tag"])
    if args.fresh:
        for f in glob.glob(ckpt + "_*.npz"):
            os.remove(f)
    log(f"task {args.task} {task}: n_modes={len(modes)} sites={len(model.basis)} "
        f"E0={E0:.10f} M_gs={M_gs}")
    t0 = time.time()
    t, C, w = Q.hole_correlation_td(model, mps, E0, p, dt=DT, t_max=T_MAX, m_max=task["M"],
                                    method=METHOD, two_way=TWO_WAY, ckpt=ckpt, log=log)
    meta = dict(task, n_modes=len(modes), n_sites=len(model.basis), E0=E0, M_gs=M_gs,
                weight=w, method=METHOD, two_way=TWO_WAY, dt=DT, t_max=T_MAX, eta=ETA,
                wall_s=time.time() - t0)
    Q.dumps_json(meta, ckpt + ".json")
    np.savez(ckpt + "_final.npz", t=t, C=C, **{k: v for k, v in meta.items()
                                                if isinstance(v, (int, float))})


# ---------------------------------------------------------------------------
def peaks(omegas_ev, A, n=2):
    """The n highest local maxima of A, as (omega_eV, height_1/eV) sorted by energy, descending."""
    i = np.where((A[1:-1] > A[:-2]) & (A[1:-1] >= A[2:]))[0] + 1
    i = i[np.argsort(-A[i])][:n]
    return sorted(((float(omegas_ev[j]), float(A[j])) for j in i), reverse=True)


def stage_analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Q = import_q()
    tddir = os.path.join(args.workdir, "td")
    runs = {}
    for f in sorted(glob.glob(os.path.join(tddir, "*_final.npz"))):
        tag = os.path.basename(f)[: -len("_final.npz")]
        d = np.load(f)
        runs[tag] = dict(t=d["t"], C=d["C"], meta=load_json(os.path.join(tddir, tag + ".json")))
    if "ref" not in runs:
        raise SystemExit("reference run td/ref_final.npz missing")
    om = OMEGAS_EV / HA2EV

    def spec(r):
        return Q.spectrum_from_correlation(r["t"], r["C"], om, ETA) / HA2EV     # 1/eV

    A_ref = spec(runs["ref"])
    lines = [f"# Convergence of A_k0(w), dRPA@HF quasiboson model", "",
             f"Reference: {runs['ref']['meta']}", "",
             "| ladder | run | max abs(A-A_ref) (1/eV) | rel | int A dw | weight | peaks (eV, 1/eV) |",
             "|---|---|---|---|---|---|---|"]

    def row(ladder, name, A, meta):
        dA = float(np.max(np.abs(A - A_ref)))
        lines.append(f"| {ladder} | {name} | {dA:.2e} | {dA / A_ref.max():.2e} | "
                     f"{np.trapz(A, OMEGAS_EV):.4f} | {meta.get('weight', float('nan')):.4f} | "
                     + ", ".join(f"({w:.2f}, {a:.3f})" for w, a in peaks(OMEGAS_EV, A)) + " |")

    row("ref", "ref", A_ref, runs["ref"]["meta"])
    for tag, r in runs.items():
        if tag != "ref":
            row(r["meta"]["ladder"], tag, spec(r), r["meta"])
    with open(os.path.join(args.workdir, "convergence.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))

    m = runs["ref"]["meta"]
    np.savetxt(os.path.join(args.workdir, "A_k0_td.dat"), np.c_[OMEGAS_EV, A_ref],
               header=f"dRPA@HF k=0 removal, eta=0.2 eV, M={m['M']} nbas={m['nb']} "
                      f"n_modes={m['n_modes']} dt={m['dt']} T={m['t_max']} E0={m['E0']:.10f} Ha "
                      f"weight={m['weight']:.6f}\nomega(eV, absolute)  A(1/eV)")

    for ladder in ["M", "nb"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(OMEGAS_EV, A_ref, "k", lw=1.5, label=f"ref (M={m['M']}, nbas={m['nb']})")
        for tag, r in runs.items():
            if r["meta"]["ladder"] == ladder:
                ax.plot(OMEGAS_EV, spec(r), lw=1, label=tag)
        ax.set_xlabel(r"$\omega$ (eV)"); ax.set_ylabel(r"$A_{k=0}(\omega)$ (1/eV)")
        ax.set_title(f"dRPA@HF: {ladder} ladder")
        ax.set_xlim(-20, -2); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(args.workdir, f"convergence_{ladder}.png"), dpi=200)
        plt.close(fig)
    done(args.workdir, "analyze")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["check", "gs", "td", "analyze", "ntasks"])
    ap.add_argument("--model", default=os.path.join(HERE, "models", "model_14e19o_k0.npz"))
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    ap.add_argument("--task", type=int, default=0, help="td array index (see --stage ntasks)")
    ap.add_argument("--fresh", action="store_true", help="td: ignore an existing checkpoint")
    a = ap.parse_args()
    if a.stage == "ntasks":
        print(len(td_tasks()))
    else:
        dict(check=stage_check, gs=stage_gs, td=stage_td, analyze=stage_analyze)[a.stage](a)
