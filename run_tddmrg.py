"""Hole spectral function A_{k=0}(omega) of the 14e19o quasiboson Hamiltonian by TD-DMRG.

One "cell" = (bath, sector):
  bath   lda | hf      dRPA@LDA or dRPA@HF plasmons (models/model_14e19o_k0_<bath>.npz)
  sector full          all 19 k-points, 14 electrons, correlated DMRG ground state
         onehole       only the 7 occupied k-points: GS = |HF>|0_b> exactly, N-1 = one hole
                       hopping among occupied momenta (the sector dCC / SC-CE / ED were solved in)

Stages (each reads the model npz, never re-runs SCF):
  check    tiny sub-model vs dense ED: DMRG E0, TD C(t) vs Lehmann C(t) (one- and two-way),
           A(w) vs Lorentzian sum; then a timing probe on the cell's full model
                                                          -> check_report.json, timing.json
  gs       ground state: bond dimension M nested inside boson levels nbas
                                                          -> gs_scan.dat, gs_converged.json, gs_nb*_M*.npz
  td       one array task per point of the convergence ladders (see td_tasks)
                                                          -> td/<tag>_C.npz, td/<tag>.json
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

# ---- time-evolution reference point and ladders (one parameter varied at a time)
DT = 0.5                                  # a.u.; TDVP step
T_MAX = 700.0                             # a.u.; e^{-eta T} = 6e-3
T_CUTS = [300.0, 500.0, 700.0]            # A(w) re-evaluated on [0, T] to show T convergence
TWO_WAY = True
METHOD = "tdvp_ps"
M_REF = 256                               # TD bond dimension
NB_REF = 4                                # n_max = 3 (one-hole ED converged to 1e-4 eV at n_max 3-4)
M_TD = [64, 128, 256, 512]
NB_TD = [2, 3, 4, 5, 6]
MODES_G2 = [6, 12, 24, 48, 96]            # keep the n largest-|g|^2 modes (whole +-q pairs)
QSHELLS = [1, 2, 3, 4]                    # keep the n smallest |q|^2 shells
DT_CHECK = (0.25, 100.0)                  # (dt, t_max) compared against the reference prefix

OMEGAS_EV = np.arange(-20.0, 0.0 + 1e-9, 0.01)
# one-hole-sector exact diagonalization of the same H (dcc_qb.exact_removal, n_max>=4)
ONEHOLE_ED_EV = dict(lda=dict(qp=-6.166, sat=-11.244), hf=dict(qp=-7.06, sat=-16.09))


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


def cell_inputs(Q, args, how="all", n=None):
    """kpts, eps, modes (sector-restricted, pruned, truncated), nelec_total, n_all_modes."""
    kpts, eps_hf, modes, nelec = Q.load_inputs(args.model)
    ks = Q.sector_kpts(kpts, eps_hf, nelec[0], args.sector)
    modes = Q.prune_modes(modes, ks)
    n_all = len(modes)
    modes = Q.select_modes(modes, how, n)
    return ks, eps_hf, modes, sum(nelec), n_all


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

    # --- timing probe on this cell's model (all modes, nbas=3)
    ks, eps, cmodes, ne_tot, _ = cell_inputs(Q, args)
    model, info = Q.build_model(ks, eps, cmodes, nbas_boson=3)
    h = Mpo(model)
    tm = dict(n_sites=len(model.basis), n_modes=len(cmodes), n_fermion_sites=2 * len(ks),
              q_shells=Q.q_shells(cmodes))
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
                              method=METHOD, two_way=True, log=lambda s: None)
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
    ks, eps, modes, ne, _ = cell_inputs(Q, args)
    hf_energy = 2 * sum(sorted(eps[k] for k in ks)[: ne // 2])
    os.makedirs(args.workdir, exist_ok=True)
    scan = open(os.path.join(args.workdir, "gs_scan.dat"), "a")
    scan.write("# nb  M  E0(Ha)  dE_M  N  n_k0up  top_level_occ  sweeps  dmrg_conv  wall_s\n")
    per_nb = {}
    E_prev_nb, m_start, nb_conv = None, 0, None
    for nb in NB_LIST:
        model, info = Q.build_model(ks, eps, modes, nbas_boson=nb)
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
    """Model-independent list: the reference point, then one-parameter-at-a-time ladders.
    Tasks whose truncation keeps every mode are skipped at run time."""
    ref = dict(M=M_REF, nb=NB_REF, how="all", n=None, dt=DT, t_max=T_MAX)
    tasks = [dict(ref, tag="ref", ladder="ref")]
    tasks += [dict(ref, M=M, tag=f"M{M}", ladder="M") for M in M_TD if M != M_REF]
    tasks += [dict(ref, nb=nb, tag=f"nb{nb}", ladder="nb") for nb in NB_TD if nb != NB_REF]
    tasks += [dict(ref, how="g2", n=n, tag=f"g2_{n}", ladder="modes_g2") for n in MODES_G2]
    tasks += [dict(ref, how="qshell", n=n, tag=f"qshell{n}", ladder="modes_qshell") for n in QSHELLS]
    tasks += [dict(ref, dt=DT_CHECK[0], t_max=DT_CHECK[1], tag=f"dt{DT_CHECK[0]}", ladder="dt")]
    return tasks


def td_ground_state(Q, args, model, info, h, nb, how, ne):
    """Reuse the gs-stage MPS when the task has all modes and that nb was scanned;
    otherwise (truncated bath or new nb) run DMRG at the converged gs bond dimension."""
    from renormalizer import Mps
    gs = load_json(os.path.join(args.workdir, "gs_converged.json"))
    g = gs["per_nb"].get(str(nb))
    if how == "all" and g is not None:
        return Mps.load(model, g["file"]), g["M"]
    M = gs["M"]
    en, mps, conv = Q.dmrg_ground_state(model, ne, M=M, mpo=h, e_rtol=0, e_atol=TOL_M / 10)
    if not conv:
        log(f"WARNING: GS for this task not converged at M={M}")
    return mps, M


def stage_td(args):
    Q = import_q()
    from renormalizer import Mpo
    task = td_tasks()[args.task]
    ks, eps, modes, ne, n_all = cell_inputs(Q, args, task["how"], task["n"])
    if task["how"] != "all" and len(modes) >= n_all:
        log(f"task {task['tag']}: keeps all {n_all} modes, identical to ref -> skipped")
        return
    tddir = os.path.join(args.workdir, "td")
    os.makedirs(tddir, exist_ok=True)
    model, info = Q.build_model(ks, eps, modes, nbas_boson=task["nb"])
    h = Mpo(model)
    mps, M_gs = td_ground_state(Q, args, model, info, h, task["nb"], task["how"], ne)
    E0 = Q.gs_energy(model, mps, h)
    p = info["orb_index"][(K0, 0)]
    ckpt = os.path.join(tddir, task["tag"])
    if args.fresh:
        for f in glob.glob(ckpt + "_*.npz"):
            os.remove(f)
    log(f"task {args.task} {task}: n_modes={len(modes)}/{n_all} sites={len(model.basis)} "
        f"E0={E0:.10f} M_gs={M_gs}")
    t0 = time.time()
    t, C, w = Q.hole_correlation_td(model, mps, E0, p, dt=task["dt"], t_max=task["t_max"],
                                    m_max=task["M"], method=METHOD, two_way=TWO_WAY,
                                    ckpt=ckpt, log=log)
    meta = dict(task, n_modes=len(modes), n_modes_all=n_all, n_sites=len(model.basis),
                q_shells=Q.q_shells(modes), E0=E0, M_gs=M_gs, weight=w, method=METHOD,
                two_way=TWO_WAY, eta=ETA, wall_s=time.time() - t0)
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

    def spec(r, T=None):
        t, C = r["t"], r["C"]
        if T is not None:
            t, C = t[t <= T + 1e-9], C[t <= T + 1e-9]
        return Q.spectrum_from_correlation(t, C, om, ETA) / HA2EV     # 1/eV

    A_ref = spec(runs["ref"])
    ref_pk = peaks(OMEGAS_EV, A_ref)
    lines = [f"# Convergence of A_k0(w), cell bath={args.bath} sector={args.sector}", "",
             f"Reference: {runs['ref']['meta']}", "",
             "| ladder | run | n_modes | max abs(A-A_ref) (1/eV) | rel | int A dw | weight | peaks (eV, 1/eV) |",
             "|---|---|---|---|---|---|---|---|"]

    def row(ladder, name, A, meta):
        dA = float(np.max(np.abs(A - A_ref)))
        lines.append(f"| {ladder} | {name} | {meta.get('n_modes', '')} | {dA:.2e} | "
                     f"{dA / A_ref.max():.2e} | {np.trapz(A, OMEGAS_EV):.4f} | "
                     f"{meta.get('weight', float('nan')):.4f} | "
                     + ", ".join(f"({w:.2f}, {a:.3f})" for w, a in peaks(OMEGAS_EV, A)) + " |")

    for T in T_CUTS:
        row("T", f"T={T:g}", spec(runs["ref"], T), runs["ref"]["meta"])
    for tag, r in runs.items():
        if tag != "ref" and r["meta"]["ladder"] != "dt":
            row(r["meta"]["ladder"], tag, spec(r), r["meta"])
    tag_dt = f"dt{DT_CHECK[0]}"
    if tag_dt in runs:
        r, rr = runs[tag_dt], runs["ref"]
        common = np.intersect1d(np.round(r["t"], 9), np.round(rr["t"], 9))
        c1 = r["C"][np.isin(np.round(r["t"], 9), common)]
        c0 = rr["C"][np.isin(np.round(rr["t"], 9), common)]
        lines += ["", f"dt check: max |C_dt{DT_CHECK[0]} - C_dt{DT}| over t <= {common[-1]:g} "
                      f"= {np.max(np.abs(c1 - c0)):.2e}"]
    if args.sector == "onehole":
        tgt = ONEHOLE_ED_EV[args.bath]
        lines += ["", f"one-hole ED of the same H: QP {tgt['qp']} eV, satellite {tgt['sat']} eV; "
                      f"TD-DMRG ref peaks: {ref_pk}"]
    with open(os.path.join(args.workdir, "convergence.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))

    m = runs["ref"]["meta"]
    np.savetxt(os.path.join(args.workdir, "A_k0_td.dat"), np.c_[OMEGAS_EV, A_ref],
               header=f"bath={args.bath} sector={args.sector} k=0 removal, eta=0.2 eV, "
                      f"M={m['M']} nbas={m['nb']} n_modes={m['n_modes']} dt={m['dt']} "
                      f"T={m['t_max']} E0={m['E0']:.10f} Ha weight={m['weight']:.6f}\n"
                      "omega(eV, absolute)  A(1/eV)")

    for ladder in ["M", "nb", "modes_g2", "modes_qshell", "T"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(OMEGAS_EV, A_ref, "k", lw=1.5, label="ref")
        if ladder == "T":
            for T in T_CUTS[:-1]:
                ax.plot(OMEGAS_EV, spec(runs["ref"], T), lw=1, label=f"T={T:g}")
        for tag, r in runs.items():
            if r["meta"]["ladder"] == ladder:
                ax.plot(OMEGAS_EV, spec(r), lw=1, label=f"{tag} ({r['meta']['n_modes']} modes)")
        ax.set_xlabel(r"$\omega$ (eV)"); ax.set_ylabel(r"$A_{k=0}(\omega)$ (1/eV)")
        ax.set_title(f"{args.bath} / {args.sector}: {ladder} ladder")
        ax.set_xlim(-20, -2); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(args.workdir, f"convergence_{ladder}.png"), dpi=200)
        plt.close(fig)
    done(args.workdir, "analyze")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["check", "gs", "td", "analyze", "ntasks"])
    ap.add_argument("--bath", choices=["lda", "hf"], default="lda")
    ap.add_argument("--sector", choices=["full", "onehole"], default="full")
    ap.add_argument("--model", default=None, help="default models/model_14e19o_k0_<bath>.npz")
    ap.add_argument("--workdir", default=None, help="default work/<bath>_<sector>")
    ap.add_argument("--task", type=int, default=0, help="td array index (see --stage ntasks)")
    ap.add_argument("--fresh", action="store_true", help="td: ignore an existing checkpoint")
    a = ap.parse_args()
    a.model = a.model or os.path.join(HERE, "models", f"model_14e19o_k0_{a.bath}.npz")
    a.workdir = a.workdir or os.path.join(HERE, "work", f"{a.bath}_{a.sector}")
    if a.stage == "ntasks":
        print(len(td_tasks()))
    else:
        dict(check=stage_check, gs=stage_gs, td=stage_td, analyze=stage_analyze)[a.stage](a)
