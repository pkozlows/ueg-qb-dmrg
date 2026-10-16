# TD-DMRG spectral functions of quasiboson Hamiltonians for the uniform electron gas

**Status: code ready, nothing has been run yet. Review requested before we submit jobs.**
See [Open questions for review](#open-questions-for-review).

## Goal

McClain et al., PRB 93, 235139 (2016), Fig. 1a, shows the k = 0 hole (removal) spectral
function of a 14-electron / 19-plane-wave uniform electron gas cell (r_s = 4, Lorentzian
broadening η = 0.2 eV). It compares CCSD, CCSDT, DMRG on the full Coulomb Hamiltonian,
HF+GW, LDA+GW and LDA+GW+C.

This repo computes the **exact** k = 0 removal spectrum of two **quasiboson Hamiltonians**
for the same cell with real-time TD-DMRG, and overlays it on Fig. 1a. The quasiboson
Hamiltonian is the one that GW (second order in g) and GW+cumulant (frozen hole) approximate.
Solving it exactly separates "error of the Hamiltonian" from "error of the approximate solution".

## Model

The Hamiltonian always has the form

H_qb = H_el + H_pl + H_ep

with

- H_el = Σ_{k,σ} ε_k^HF  a†_{kσ} a_{kσ},  where ε_k^HF = k²/2 + Σ^x_k − n_k |v_M|
- H_pl = Σ_{ν,q} Ω_{νq}  b†_{νq} b_{νq}
- H_ep = Σ_{ν,q} Σ_{k,σ} g_ν(q)  a†_{k+q,σ} a_{kσ} ( b†_{ν,−q} + b_{νq} )

Here n_k is the HF occupation and v_M is the Madelung constant. ε^HF is the probe-charge
shifted-Ewald HF eigenvalue (`hfes_prew`): occupied levels carry −|v_M|, which is the
Koopmans / charged-excitation convention used for the absolute ω axis of McClain Fig. 1a.

The plasmons (ν labels the dRPA eigenmodes in each momentum-transfer sector q) and their
couplings come from a per-q direct-RPA (Casida) calculation. There are two choices:

| bath | Ω_{νq}, g_ν(q) computed from | file |
|---|---|---|
| `lda` | dRPA@LDA: LDA orbital energies | `models/model_14e19o_k0_lda.npz` |
| `hf`  | dRPA@HF: `hfes_prew` | `models/model_14e19o_k0_hf.npz` |

g_ν(q) = |V^ν_{k,k−q}(q)| is independent of k for plane-wave orbitals.

**Madelung caveat for the `hf` bath (deliberate).** Because only the occupied `hfes_prew`
carry −|v_M|, the particle-hole gaps ε_a − ε_i entering dRPA@HF are one Madelung constant
(4.97 eV) too high. The dRPA@HF plasmon therefore sits near 10.6 eV instead of about 4.7 eV.
We use the bath exactly as defined above (`hfes_prew`) on purpose. The Madelung-free
alternative is in `hfes_unsh`, which is stored in the npz.

### Sectors

Each bath is solved in two sectors (`--sector`):

- **`full`**: all 19 k-points (38 spin-orbitals) with 14 electrons. The ground state is a
  correlated DMRG state, and the hole can scatter into virtual momenta.
- **`onehole`**: only the 7 occupied k-points (14 spin-orbitals).
  - On this set the 14-electron state is the filled shell, so every a†_{k+q} a_k annihilates
    it and the ground state is |HF⟩ ⊗ |0_b⟩ exactly. `gs` asserts E₀ = 2 Σ ε_occ.
  - The N−1 sector is then a single hole hopping among occupied momenta while emitting and
    absorbing plasmons.
  - This is the sector in which our earlier coupled-cluster / self-consistent cumulant /
    exact-diagonalization work on the same Hamiltonian was done. The exact-diagonalization
    targets are QP −6.17 eV and satellite −11.24 eV (`lda`), and QP −7.06 eV and satellite
    −16.09 eV (`hf`).
  - It is the built-in sanity check, and it gives the comparison "one-hole vs full".

Modes that couple no pair of k-points in the sector are pruned, always as whole ±q pairs.

### Mapping to Renormalizer (`qb_dmrg.py`)

**Fermions** are Jordan–Wigner mapped onto `BasisHalfSpin` sites:

- Each site has quantum number `sigmaqn=[0,1]`, i.e. U(1) particle number.
- Site order is (k₁↑, k₁↓, k₂↑, …) in the order k-points appear in the npz.
- a_p = Z₀⋯Z_{p−1} σ⁺_p and a†_p a_q = σ⁻_p Z⋯Z σ⁺_q, with qn given explicitly.
  (Renormalizer's `sigma_+` lowers 1 → 0 in this basis.)

**Bosons** are `BasisSHO` sites truncated at `nbas` levels, so n_max = nbas − 1.

**Site ordering.** Bosons are interleaved among the fermion sites with a uniform stride
(`interleave=True`); see the open questions.

## Spectral function

The hole spectral function is

A_k(ω) = Σ_n |⟨n|a_{k↑}|0⟩|² δ_η(ω − (E₀ − E_n^{N−1}))

where δ_η is a Lorentzian with half-width at half-maximum η = 0.2 eV. ω is the absolute
removal energy, the same frame as McClain Fig. 1a.

### Time-domain route (`hole_correlation_td`)

1. |φ₀⟩ = a_p|0⟩ is normalised. Its norm² ⟨n_p⟩ is kept as the total spectral weight.
2. The MPO carries `offset=E₀`, so TDVP evolves with e^{−i(H−E₀)t} and no phase bookkeeping
   is needed. (In Renormalizer 0.0.11, `Mps.evolve(mpo, dt)` with real dt applies e^{−iH dt}.)
3. The correlation function is
   C(t) = ⟨φ₀| e^{−i(H−E₀)t} |φ₀⟩ = Σ_n w_n e^{+iω_n t},  with ω_n = E₀ − E_n.
4. **Two-way propagation (default).** H, |0⟩ and a_p are real, so
   C(t) = φ(t/2)ᵀ φ(t/2), with φ(τ) = e^{−i(H−E₀)τ} φ₀.
   - This is a bilinear product with no complex conjugation (`Mps.dot` does not conjugate).
   - One evolution to T/2 therefore gives C on [0, T], with sample spacing 2·dt.
   - The MPS only has to carry the entanglement generated by time T/2.
   - `two_way=False` gives the textbook ⟨φ(0)|φ(t)⟩. The `check` stage validates both
     against an exact Lehmann sum.
5. The spectrum is
   A(ω) = (1/π) Re ∫₀^T dt e^{iωt} C*(t) e^{−ηt}  (trapezoid rule),
   which tends to the Lorentzian sum above as T → ∞.

### How long and how fine a time grid?

- **Total time T.** The broadening sets T, not the frequency spacing you want on the plot.
  After damping, the integrand is e^{−ηT} at the end of the window, and that sets the
  truncation ringing. With η = 0.2 eV = 7.35·10⁻³ Ha:

  | T (a.u.) | e^{−ηT} (ringing) |
  |---|---|
  | 300 | 11 % |
  | 410 (= 3/η) | 5 % |
  | 500 | 2.5 % |
  | 700 (≈ 5/η) | 0.6 % |

  Going beyond about 5/η buys nothing. You can plot A(ω) on as fine an ω grid as you like,
  because the transform is evaluated directly, not by FFT.
- **Time step dt.** dt is set by TDVP accuracy, not by Nyquist. The removal poles span
  roughly −20 to 0 eV (|ω| < 0.75 Ha), while sampling at 2·dt = 1 a.u. resolves |ω| up to
  π Ha. We use dt = 0.5 a.u. and check it against dt = 0.25 over t ≤ 100.
- **Reference point.** dt = 0.5 and T = 700 means 700 TDVP steps with two-way propagation.
  The earlier cumulant grid in `pyqcpbc_chk.py` (T = 628, dt = 0.52) was about right.

## Convergence ladders (`run_tddmrg.py`)

**Ground state (`gs`).** Bond dimension M ∈ {32 … 1024} is scanned inside boson levels
nbas ∈ {1 … 6}. The scan stops when |ΔE₀| < 10⁻⁶ Ha, the summed top-level boson occupation
is < 10⁻⁶, and DMRG's own sweep test passes.

**Time evolution (`td`).** One SLURM array task per point; one parameter is varied at a time
around the reference.

- Reference: M = 256, nbas = 4, all modes, 1-site TDVP (`tdvp_ps`), dt = 0.5, T = 700.
- Ladders:

| ladder | values | question |
|---|---|---|
| `M` | 64, 128, 256, 512 | TD bond dimension |
| `nb` | 2, 3, 4, 5, 6 | boson truncation. The n-th plasmon replica needs n_max ≥ n, so the ground-state test alone is not enough. |
| `modes_g2` | keep the 6, 12, 24, 48, 96 largest-\|g\|² modes | **can the plasmon bath be truncated?** Our expectation is that every mode is needed. |
| `modes_qshell` | keep \|q\|² shells 1, ≤2, ≤3, ≤4 | same question, organised by momentum transfer |
| `T` | A(ω) recomputed on [0, 300], [0, 500], [0, 700] | time-window truncation |
| `dt` | 0.25 vs 0.5 over t ≤ 100 | TDVP time-step error |

- Truncated-bath tasks recompute the ground state for their own bath.
- A truncation that keeps every mode is skipped as identical to the reference.

**Analysis (`analyze`).** For every run, `convergence.md` lists:

- max |A − A_ref|, absolute and relative to max A_ref;
- ∫A dω compared with ⟨n_k0⟩;
- the two largest peaks (QP and satellite).

It also writes one figure per ladder and `A_k0_td.dat` (the reference spectrum).

### Built-in validation (`check`)

The `check` stage runs on a tiny sub-model: k ∈ {0, ±x̂}, two electrons, and the two
strongest ±x̂ plasmon pairs. It compares against dense exact diagonalization:

- **Ground-state energy:** DMRG E₀ vs exact, asserted to 10⁻⁹ Ha.
- **TD correlation function:** C(t) vs the exact Lehmann sum at the production dt, for both
  one-way and two-way propagation, asserted to 10⁻⁶. M = 128 exceeds the exact bond
  dimension here, so any residual is TDVP error.
- **Spectrum:** A(ω) error is split into a TDVP part and a finite-T part.
- **HF limit:** with nbas = 1 there must be a single pole at ε_k0 with unit weight.

It then times a ground-state sweep and a TDVP step on the cell's real model and writes
`timing.json`, which is used to size the production jobs.

## Running

```bash
# 0. environment: renormalizer 0.0.11, numpy, scipy, matplotlib (GPU: + cupy-cuda12x)
# 1. model inputs (needs the group's private pyqcpbc/ueg stack; skip if models/*.npz exist)
python generate_model.py --bath lda
python generate_model.py --bath hf
# 2. everything else, as a SLURM chain per cell (setup -> check -> gs -> td array -> analyze)
bash submit.sh --dry                     # print the sbatch commands
bash submit.sh                           # all four cells on the joonholee partition
bash submit.sh --cells "lda_onehole" --gpu
# 3. the figure
python plot_fig1a.py                     # -> fig1a_qb_dmrg.png
```

Single stages run directly, e.g.
`python run_tddmrg.py --stage check --bath lda --sector onehole`.

- `td` tasks checkpoint every 50 steps (`work/<cell>/td/<tag>_{C,mps}.npz`) and resume
  automatically; pass `--fresh` to restart a task.
- `submit.sh` skips any stage whose `work/<cell>/<stage>.done` exists.

## Files

| file | content |
|---|---|
| `qb_dmrg.py` | model construction, sector/mode selection, DMRG ground state, TD correlation function, transform, dense exact diagonalization |
| `run_tddmrg.py` | stages `check`, `gs`, `td`, `analyze` and the convergence ladders |
| `generate_model.py` | builds the model npz (HF SCF, LDA eigenvalues, per-q dRPA) |
| `submit.sh` | SLURM chain |
| `plot_fig1a.py` | McClain Fig. 1a with our spectra overlaid |
| `models/` | model inputs (see `models/README.md`) |
| `reference/claude_fig1a_*.csv` | McClain Fig. 1a curves, vector-extracted from the published PDF (absolute ω in eV, A in 1/eV; the header of each file gives the provenance) |

## Open questions for review

1. **Site ordering.** Bosons are interleaved with a fixed stride, and fermions are in npz
   k-order. Would a better ordering lower the bond dimension significantly? Candidates:
   order k by |k| or ε, and place each (ν, ±q) pair next to the k-shells it connects.
2. **Symmetries.** Only U(1) particle number is used. Total momentum (the hole has K = 0)
   and S_z are also conserved. Is it worth encoding them as extra quantum numbers in
   Renormalizer?
3. **Integrator.** We use 1-site TDVP (`tdvp_ps`) at fixed M after `expand_bond_dimension`.
   Would you start with `tdvp_ps2` (2-site) for the first few steps, or use it throughout?
   Is dt = 0.5 a.u. reasonable here?
4. **Two-way trick.** Does C(t) = φ(t/2)ᵀφ(t/2) hold as implemented? We rely on a real H,
   a real ground state and `Mps.dot` being bilinear. The `check` stage tests it on the tiny
   model.
5. **Boson truncation.** The ground-state test (top-level occupation) underestimates what the
   satellite needs. Is the nbas ladder in `td` the right way to settle it?
6. **Mode truncation.** If the bath can be truncated, the `modes_g2` / `modes_qshell` ladders
   should show it. Is there a smarter criterion?
7. **Madelung in the dRPA@HF bath.** This is deliberate (see the Model section), but worth
   a second opinion.
