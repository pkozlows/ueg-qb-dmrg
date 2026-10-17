#!/bin/bash
# TD-DMRG A_{k=0}(w) of the 14e19o quasiboson Hamiltonians on FASRC (SLURM).
#   setup (generate_model.py) -> check -> gs -> td (array, one ladder point per task) -> analyze,
#   chained with afterok. A stage is skipped if <workdir>/<stage>.done exists (or, for setup, if
#   the model npz exists); delete it to force a rerun.
#
# usage:  bash submit.sh [--scheme shifted|standard] [--from check|gs|td|analyze]
#                        [--gpu] [--dry]
#   --scheme  Madelung scheme (README "Two Madelung schemes"). shifted (default): shifted Ewald
#             throughout, models/model_14e19o_k0.npz, work/. standard: standard Ewald + shift at
#             the end, models/model_14e19o_k0_standard.npz, work_standard/, job names qbdmrg_std_*
#   --gpu  run gs/td on one A100 (needs `pip install cupy-cuda12x` in the env; Renormalizer
#          then offloads the Krylov/contraction work, SVDs stay on the CPU)
#   --dry  print the sbatch commands instead of submitting
set -euo pipefail
WD=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/n/home03/pkozlowski/.conda/envs/mcgw_env/bin/python}
FROM=check; GPU=0; DRY=0; SCHEME=shifted
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) FROM=$2; shift 2;;
        --scheme) SCHEME=$2; shift 2;;
        --gpu) GPU=1; shift;;
        --dry) DRY=1; shift;;
        *) echo "unknown arg $1"; exit 1;;
    esac
done
order=(check gs td analyze)
start=0; for i in "${!order[@]}"; do [[ ${order[$i]} == "$FROM" ]] && start=$i; done
case $SCHEME in
    shifted)  MODEL=$WD/models/model_14e19o_k0.npz; W=$WD/work; PFX=""; SARG="";;
    standard) MODEL=$WD/models/model_14e19o_k0_standard.npz; W=$WD/work_standard; PFX="std_"
              SARG=" --scheme standard";;
    *) echo "unknown scheme $SCHEME"; exit 1;;
esac
NT=$($PY $WD/run_tddmrg.py --stage ntasks)
mkdir -p "$WD/logs"

cpu=(-p joonholee -n 1 --cpus-per-task=32
     --export=ALL,OMP_NUM_THREADS=32,OPENBLAS_NUM_THREADS=32,MKL_NUM_THREADS=32,MPLBACKEND=Agg)
if (( GPU )); then
    heavy=(-p gpu --gres=gpu:1 -n 1 --cpus-per-task=16
           --export=ALL,RENO_GPU=0,OMP_NUM_THREADS=16,MKL_NUM_THREADS=16,MPLBACKEND=Agg)
    pre="module load cuda/12.4.1-fasrc01 && "; heavy_t=3-00:00:00
else
    heavy=("${cpu[@]}"); pre=""; heavy_t=7-00:00:00
fi

sb() {   # job name, dependency array name, sbatch args... -- command ; echoes job id
    local name=$1 depvar=$2; shift 2
    local -n _d=$depvar
    local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
    name=$PFX$name
    local cmd=(sbatch --parsable -D "$WD" "${_d[@]}" -J "qbdmrg_$name"
               -o "$WD/logs/${name}_%A_%a.out" -e "$WD/logs/${name}_%A_%a.err" "${args[@]}" --wrap "$*")
    if (( DRY )); then echo "${cmd[*]}" >&2; echo "DRY_$name"; return; fi
    local id; id=$("${cmd[@]}"); echo "${id%%;*}"
}

dep=()
if [[ ! -f $MODEL ]]; then
    nodep=()
    id=$(sb setup nodep "${cpu[@]}" --mem=50G -t 4:00:00 -- "$PY $WD/generate_model.py$SARG")
    echo "setup -> $id"; dep=(--dependency=afterok:$id)
fi

mkdir -p "$W"
R="$PY $WD/run_tddmrg.py$SARG"
for i in "${!order[@]}"; do
    stage=${order[$i]}
    if (( i < start )) || [[ -f $W/$stage.done ]]; then echo "skip $stage"; continue; fi
    case $stage in
        check)   id=$(sb check dep "${cpu[@]}" --mem=100G -t 12:00:00 -- "$R --stage check");;
        gs)      id=$(sb gs dep "${heavy[@]}" --mem=180G -t $heavy_t -- "$pre$R --stage gs");;
        td)      id=$(sb td dep "${heavy[@]}" --mem=96G -t $heavy_t --array=0-$((NT-1)) -- \
                      "$pre$R --stage td --task \$SLURM_ARRAY_TASK_ID")
                 dep=(--dependency=afterany:${id%%_*});;   # analyze runs on whatever finished
        analyze) id=$(sb analyze dep "${cpu[@]}" --mem=20G -t 1:00:00 -- "$R --stage analyze");;
    esac
    echo "$stage -> $id"
    [[ $stage != td ]] && dep=(--dependency=afterok:$id)
done
