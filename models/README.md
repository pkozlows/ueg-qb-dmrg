# Model inputs

`generate_model.py` writes one file per Madelung scheme (HF band, dRPA@HF plasmons):
`model_14e19o_k0.npz` (shifted Ewald, `--scheme shifted`), `model_14e19o_k0_standard.npz`
(standard Ewald + end shift, `--scheme standard`) and `model_14e19o_k0_selfenergy.npz`
(shifted Ewald for the self-energy only, `--scheme selfenergy`). That script needs the group's private
electronic-structure stack. Once the files exist they are committed
here, so the DMRG side runs with only Renormalizer, NumPy and SciPy.

Cell: 14 electrons, 19 plane waves, r_s = 4.

## Contents

All energies are in hartree.

| key | shape | meaning |
|---|---|---|
| `kpts` | (19, 3) int | plane-wave momenta in units of 2π/L |
| `eps` | (19,) | F_k of H_el in `kpts` order: `hfes_prew` (shifted, selfenergy) or `hfes_unsh` (standard) |
| `mode_nu`, `mode_q` | (n_modes,), (n_modes, 3) | plasmon label ν and momentum transfer q; every q appears together with −q |
| `mode_Omega` | (n_modes,) | Ω_{νq} |
| `mode_g` | (n_modes,) | g_ν(q) = \|V^ν_{k,k−q}(q)\|, independent of k |
| `nelec` | (2,) | (7, 7) |
| `bath` | str | `drpa@hf` (the eigenvalues that entered dRPA) |
| `scheme` | str | `shifted`, `standard` or `selfenergy` (absent in the first shifted file = shifted) |
| `omega_shift_ha` | scalar | shift applied to the removal spectrum at the end: v_M/2 (standard), 0 otherwise; absent = 0 |
| `hfes_prew`, `hfes_unsh` | (19,) | probe-charge HF and standard-Ewald HF eigenvalues in SCF order (`hfes_unsh` absent in the first shifted file) |
| `madelung_xi` | scalar | Madelung constant ξ of the cell |

Read the file with `qb_dmrg.load_inputs`.
