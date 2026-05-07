#!/bin/bash

set -e

# Defaults
DATA_TERM=""
POLY_DEGREES=(1 2 3)
USE_GENETIC=false
USE_SGD=false
NUM_POINTS=20
NUM_RESTARTS=5
EXTERNALITY_COST=0.01
TAG="test"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Runs the collateralized auction optimizer on CSV files in data/samples/.
By default uses grid search on all files with polynomial degrees 1, 2, and 3.

Options:
  --data <pattern>           Only run files whose name starts with <pattern>
                             (e.g. "a" matches a_single_modal_normal_distribution.csv)
                             Default: all CSV files in data/samples/
  --polynomial-degree <deg>  Run only this degree instead of all three (1, 2, 3)
  --genetic                  Use genetic algorithm instead of grid search
  --sgd                      Use SGD optimizer instead of grid search
  --num-points <n>           Grid points per coefficient dimension (default: 20, grid search only)
                             Grid size = n^(degree+1); reduce for higher degrees
  --num-restarts <n>         SGD restarts sweeping intercept across v range (default: 5, SGD only)
  --externality-cost <c>     Externality cost parameter (default: 0.01)
  --tag <tag>                Label appended to run ID: <file-prefix>_<degree>_<tag>
                             (default: test)
  --help                     Show this message and exit

Fixed parameters (edit script to change):
  --k 1  --seed 1234

Examples:
  $(basename "$0")
  $(basename "$0") --data a --polynomial-degree 2
  $(basename "$0") --genetic --tag run1
  $(basename "$0") --data b --num-points 15 --tag sweep
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)             DATA_TERM="$2";       shift 2 ;;
        --polynomial-degree) POLY_DEGREES=("$2"); shift 2 ;;
        --genetic)          USE_GENETIC=true;     shift   ;;
        --sgd)              USE_SGD=true;         shift   ;;
        --num-points)       NUM_POINTS="$2";      shift 2 ;;
        --num-restarts)     NUM_RESTARTS="$2";       shift 2 ;;
        --externality-cost) EXTERNALITY_COST="$2";  shift 2 ;;
        --tag)              TAG="$2";               shift 2 ;;
        --help)             usage ;;
        *) echo "Unknown argument: $1"; echo "Run $(basename "$0") --help for usage."; exit 1 ;;
    esac
done

# Collect matching files
if [[ -z "$DATA_TERM" ]]; then
    files=(data/samples/*.csv)
else
    files=(data/samples/${DATA_TERM}*.csv)
fi

if [[ ${#files[@]} -eq 0 || ! -e "${files[0]}" ]]; then
    echo "No CSV files found matching pattern '${DATA_TERM}*.csv' in data/samples/"
    exit 1
fi

mkdir -p output/results

for file in "${files[@]}"; do
    filename=$(basename "$file")
    datatag="${filename:0:1}"

    for poly in "${POLY_DEGREES[@]}"; do
        id="${datatag}_${poly}_${TAG}"
        echo "=========================================="
        echo "data=$filename  degree=$poly  id=$id"
        echo "=========================================="

        start=$(date +%s)

        if $USE_GENETIC; then
            python Collateralized_Auction_genetic_script.py \
                --data "$file" \
                --k 1 \
                --externality-cost "$EXTERNALITY_COST" \
                --polynomial-degree "$poly" \
                --seed 1234 \
                --id "$id"
        elif $USE_SGD; then
            python Collateralized_Auction_sgd.py \
                --data "$file" \
                --k 1 \
                --externality-cost "$EXTERNALITY_COST" \
                --polynomial-degree "$poly" \
                --seed 1234 \
                --num-restarts "$NUM_RESTARTS" \
                --id "$id"
        else
            python Collateralized_Auction_grid_search.py \
                --data "$file" \
                --k 1 \
                --externality-cost "$EXTERNALITY_COST" \
                --polynomial-degree "$poly" \
                --seed 1234 \
                --num-points "$NUM_POINTS" \
                --id "$id"
        fi

        elapsed=$(( $(date +%s) - start ))
        echo "Done: $id  (${elapsed}s)"
    done
done
