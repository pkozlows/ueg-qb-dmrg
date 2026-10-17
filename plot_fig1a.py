"""McClain et al. 2016 Fig 1a (14e19o, rs=4, k=0, eta=0.2 eV) with the TD-DMRG spectra of
the dRPA@HF quasiboson Hamiltonian overlaid, one curve per Madelung scheme (work/ = shifted
Ewald, work_standard/ = standard Ewald + shift). A curve whose A_k0_td.dat does not exist yet
is skipped, so this can be run while jobs are still going.

    python plot_fig1a.py [--out fig1a_qb_dmrg.png]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference")

# digitised McClain Fig 1a (absolute omega in eV, probe-charge Ewald frame; A in 1/eV)
MCCLAIN = [("ccsd", "CCSD", dict(color="#d6b9da", lw=4)),
           ("ccsdt", "CCSDT", dict(color="0.35", lw=2)),
           ("dmrg", "DMRG (full H)", dict(color="k", ls="none", marker="o", ms=4)),
           ("hfgw", "HF+GW", dict(color="#e4312b", lw=1)),
           ("ldagw", "LDA+GW", dict(color="#3a57a7", lw=1, ls="--")),
           ("ldagwc", "LDA+GW+C", dict(color="#4b9a4b", lw=1, ls="-."))]

OURS = [("work", "QB dRPA@HF, shifted Ewald (TD-DMRG)", dict(color="C4", lw=2)),
        ("work_standard", "QB dRPA@HF, standard Ewald + shift (TD-DMRG)", dict(color="C1", lw=2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fig1a_qb_dmrg.png"))
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, label, sty in MCCLAIN:
        d = np.loadtxt(os.path.join(REF, f"claude_fig1a_{name}.csv"), delimiter=",", comments="#")
        ax.plot(d[:, 0], d[:, 1], label=f"{label} (McClain)", **sty)
    for workdir, label, sty in OURS:
        f = os.path.join(HERE, workdir, "A_k0_td.dat")
        if os.path.exists(f):
            d = np.loadtxt(f)       # omega axis already includes the scheme's end shift
            ax.plot(d[:, 0], d[:, 1], label=label, **sty)
        else:
            print(f"missing {f} (skipped)")
    ax.set_xlabel(r"$\omega$ (eV)")
    ax.set_ylabel(r"$A_{k=0}(\omega)$ (1/eV)")
    ax.set_xlim(-20, -2)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
