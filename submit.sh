#!/bin/bash
# TD-DMRG A_{k=0}(w) of the 14e19o quasiboson Hamiltonians on FASRC (SLURM).
#   per bath:  setup (generate_model.py)
#   per cell (bath x sector):  check -> gs -> td (array, one ladder point per task) -> analyze
# chained with afterok. A stage is skipped if work/<cell>/<stage>.done exists (or, for setup,
# if the model npz exists); delete it to force a rerun.
#
# usage:  bash submit.sh [--cells "lda_full hf_full lda_onehole hf_onehole"] [--from check|gs|td|analyze]
#                        [--gpu] [--dry]
#   --gpu  run gs/td on one A100 (needs `pip install cupy-cuda12x` in the env; Renormalizer
#          then offloads the Krylov/contraction work, SVDs stay on the CPU)
#   --dry  print the sbatch commands instead of submitting
set -euo pipefail
WD=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/n/home03/pkozlowski/.conda/envs/mcgw_env/bin/python}
CELLS="lda_full hf_full lda_onehole hf_onehole"
FROM=check; GPU=0; DRY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --cells) CELLS=$2; shift 2;;
        --from) FROM=$2; shift 2;;
        --gpu) GPU=1; shift;;
        --dry) DRY=1; shift;;
        *) echo "unknown arg $1"; exit 1;;
    esac
done
order=(check gs td analyze)
start=0; for i in "${!order[@]}"; do [[ ${order[$i]} == "$FROM" ]] && start=$i; done
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
    local cmd=(sbatch --parsable -D "$WD" "${_d[@]}" -J "qbdmrg_$name"
               -o "$WD/logs/${name}_%A_%a.out" -e "$WD/logs/${name}_%A_%a.err" "${args[@]}" --wrap "$*")
    if (( DRY )); then echo "${cmd[*]}" >&2; echo "DRY_$name"; return; fi
    local id; id=$("${cmd[@]}"); echo "${id%%;*}"
}

declare -A setup_dep
for bath in lda hf; do
    model=$WD/models/model_14e19o_k0_$bath.npz
    nodep=()
    if [[ -f $model ]]; then setup_dep[$bath]=""; continue; fi
    id=$(sb "setup_$bath" nodep "${cpu[@]}" --mem=50G -t 4:00:00 -- \
         "$PY $WD/generate_model.py --bath $bath")
    echo "setup_$bath -> $id"; setup_dep[$bath]=$id
done

for cell in $CELLS; do
    bath=${cell%_*}; sector=${cell#*_}
    W=$WD/work/$cell; mkdir -p "$W"
    R="$PY $WD/run_tddmrg.py --bath $bath --sector $sector"
    dep=(); [[ -n ${setup_dep[$bath]} ]] && dep=(--dependency=afterok:${setup_dep[$bath]})
    for i in "${!order[@]}"; do
        stage=${order[$i]}
        if (( i < start )) || [[ -f $W/$stage.done ]]; then echo "$cell: skip $stage"; continue; fi
        case $stage in
            check)   id=$(sb "${cell}_check" dep "${cpu[@]}" --mem=100G -t 12:00:00 -- "$R --stage check");;
            gs)      id=$(sb "${cell}_gs" dep "${heavy[@]}" --mem=180G -t $heavy_t -- "$pre$R --stage gs");;
            td)      id=$(sb "${cell}_td" dep "${heavy[@]}" --mem=180G -t $heavy_t --array=0-$((NT-1)) -- \
                          "$pre$R --stage td --task \$SLURM_ARRAY_TASK_ID")
                     dep=(--dependency=afterany:${id%%_*});;   # analyze runs on whatever finished
            analyze) id=$(sb "${cell}_analyze" dep "${cpu[@]}" --mem=20G -t 1:00:00 -- "$R --stage analyze");;
        esac
        echo "$cell: $stage -> $id"
        [[ $stage != td ]] && dep=(--dependency=afterok:$id)
    done
done
