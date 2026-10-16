# Model inputs

`generate_model.py --bath {lda,hf}` writes `model_14e19o_k0_<bath>.npz`. That script needs
the group's private electronic-structure stack. Once the files exist they are committed
here, so the DMRG side runs with only Renormalizer, NumPy and SciPy.

Cell: 14 electrons, 19 plane waves, r_s = 4.

## Contents

All energies are in hartree.

| key | shape | meaning |
|---|---|---|
| `kpts` | (19, 3) int | plane-wave momenta in units of 2π/L |
| `eps` | (19,) | ε_k^HF = `hfes_prew` in `kpts` order (occupied levels carry −\|v_M\|) |
| `mode_nu`, `mode_q` | (n_modes,), (n_modes, 3) | plasmon label ν and momentum transfer q; every q appears together with −q |
| `mode_Omega` | (n_modes,) | Ω_{νq} |
| `mode_g` | (n_modes,) | g_ν(q) = \|V^ν_{k,k−q}(q)\|, independent of k |
| `nelec` | (2,) | (7, 7) |
| `bath` | str | `lda` or `hf` (which eigenvalues entered dRPA) |
| `lda_sps`, `hfes_prew`, `hfes_unsh` | (19,) | LDA, probe-charge HF and shifted-Ewald HF eigenvalues (for reference) |
| `madelung_xi` | scalar | Madelung constant ξ of the cell |

Read the file with `qb_dmrg.load_inputs`.
