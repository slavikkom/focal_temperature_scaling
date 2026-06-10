#!/bin/bash
# Usage:
#   bash cifar10c_eval_array_v2.sh help
#   bash cifar10c_eval_array_v2.sh manifest
#   bash cifar10c_eval_array_v2.sh missing-file
#   bash cifar10c_eval_array_v2.sh missing-content
#   sbatch --array=0-<missing_count_minus_1> cifar10c_eval_array_v2.sh selective-run-missing-file
#   sbatch --array=0-<missing_content_count_minus_1> cifar10c_eval_array_v2.sh selective-run-missing-content
#   sbatch --array=0-<total_jobs_minus_1> cifar10c_eval_array_v2.sh run-from-scratch
#
# Default mode is help. Edit the configuration variables below before running,
# as with the original version of this script.

#SBATCH --job-name=eval_cf10c
#SBATCH --output=slurm_logs_cifar10c/eval_job_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --nodelist=falcon1,falcon2,falcon3,falcon4,falcon5,falcon6,pegasus,pegasus2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


SCRIPT_NAME="cifar10c_eval_array_v2.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ "$(basename "$SLURM_SUBMIT_DIR")" == "evaluate_scripts" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  elif [[ -d "${SLURM_SUBMIT_DIR}/evaluate_scripts" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
    SCRIPT_DIR="${REPO_ROOT}/evaluate_scripts"
  else
    echo "Could not infer repo root from SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" >&2
    echo "Submit from the repo root or from the evaluate_scripts directory." >&2
    exit 1
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
cd "$SCRIPT_DIR"

MODE=${1:-help} # help, manifest, missing-file, missing-content, selective-run-*, or run-from-scratch

# Configurable seed list and parameter values
SEED_DIRS=(42 123 2023)
ALPHAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
BETAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
GAMMAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
# GAMMAS=(2.0 3.0 5.0)
KAPPAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
LABEL_SMOOTHING_VALUES=(0.05 0.1 0.15)

CORRUPTIONS=(
  "brightness"
  "contrast"
  "defocus_blur"
  "elastic_transform"
  "fog"
  "frost"
  "gaussian_blur"
  "gaussian_noise"
  "glass_blur"
  "impulse_noise"
  "jpeg_compression"
  "motion_blur"
  "pixelate"
  "saturate"
  "shot_noise"
  "snow"
  "spatter"
  "speckle_noise"
  "zoom_blur"
)
SEVERITIES=(1 2 3 4 5)

EPOCH=350
SMOKE_ARG="" # --smoke-test for quick check or empty string for full run
GPU_FLAG="-g"
DEBUG=false # set to true for debugging which won't run the evaluation only to debug this script by printouts

DATASET_ROOT="../Data/datasets"
SAVE_BASE="../MODEL_DIRECTORY/CIFAR10"
# EVAL_BASE="../RESULTS/CIFAR10C"
EVAL_BASE="../RESULTS/hpc_results_june26/CIFAR10C_epoch${EPOCH}_with_dirichlet"

EXPECTED_FILES_LIST="./cifar10c_expected_files.txt"
MISSING_FILES_LIST="./cifar10c_missing_files.txt"
MISSING_CONTENT_FILES_LIST="./cifar10c_missing_content_files.txt"
EVALUATE_LINKS=(softmax exp_p exp_1mp)

# Required JSON content for missing-content mode. Leave an array empty to skip
# that check. Top-level keys are checked directly on the root JSON object.
# T_DICT links are checked under the top-level "T_dict" object.
REQUIRED_TOP_LEVEL_KEYS=(dirichlet_calibrated)
REQUIRED_T_DICT_LINKS=(softmax exp_p exp_1mp)

mkdir -p "$EVAL_BASE"

# List of model patterns (comment out to exclude any)
MODELS=(
  "resnet50_brier_score"
  # "resnet50_proper_focal_loss"
  "resnet50_cross_entropy"
  "resnet50_exp_1mp"
  "resnet50_exp_p"
  # "resnet50_focal_loss_adaptive"
  "resnet50_focal_loss"
  # "resnet50_generalized_focal"
  # "resnet50_linear"
  # "resnet50_log_power"
  # "resnet50_one_minus_power"
)

# Build one array job per model/seed. Each job loops over corruptions/severities.
COMBINATIONS=()
for seed_idx in "${SEED_DIRS[@]}"; do
  for model in "${MODELS[@]}"; do
    case $model in
      "resnet50_brier_score")
        # No parameters required
        COMBINATIONS+=("$seed_idx|$model|none|none")
        ;;
      "resnet50_cross_entropy")
        # No parameters required
        COMBINATIONS+=("$seed_idx|$model|none|none")
        for smoothing in "${LABEL_SMOOTHING_VALUES[@]}"; do
          COMBINATIONS+=("$seed_idx|${model}_${smoothing}|none|none")
        done
        ;;
      "resnet50_exp_1mp"|"resnet50_exp_p")
        # Single alpha parameter
        for alpha in "${ALPHAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|alpha_$alpha|none")
        done
        ;;
      "resnet50_focal_loss_adaptive"|"resnet50_focal_loss"|"resnet50_proper_focal_loss")
        # Single gamma parameter
        for gamma in "${GAMMAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|gamma_$gamma|none")
        done
        ;;
      "resnet50_generalized_focal")
        # Both gamma and beta parameters
        for gamma in "${GAMMAS[@]}"; do
          for beta in "${BETAS[@]}"; do
            COMBINATIONS+=("$seed_idx|$model|beta_$beta|gamma_$gamma")
          done
        done
        ;;
      "resnet50_linear"|"resnet50_one_minus_power")
        # Single beta parameter
        for beta in "${BETAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|beta_$beta|none")
        done
        ;;
      "resnet50_log_power")
        # Single kappa parameter
        for kappa in "${KAPPAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|kappa_$kappa|none")
        done
        ;;
    esac
  done
done

# Calculate total jobs based on the length of the combinations list
TOTAL_JOBS=${#COMBINATIONS[@]}

build_model_name() {
  local model="$1"
  local param1="$2"
  local param2="$3"

  if [[ "$param1" == "none" && "$param2" == "none" ]]; then
    printf '%s\n' "$model"
  elif [[ "$param2" == "none" ]]; then
    printf '%s_%s\n' "$model" "$param1"
  else
    printf '%s_%s_%s\n' "$model" "$param1" "$param2"
  fi
}

result_file_name_for_model() {
  local model_name="$1"
  local result_stem="${model_name#resnet50_}"
  printf '%s_%s.json\n' "$result_stem" "$EPOCH"
}

expected_json_path() {
  local corruption="$1"
  local severity="$2"
  local seed="$3"
  local model_name="$4"
  local result_file
  result_file="$(result_file_name_for_model "$model_name")"
  printf '%s/%s-%s/%s/%s\n' "$EVAL_BASE" "$corruption" "$severity" "$seed" "$result_file"
}

emit_expected_files() {
  local entry seed_idx model param1 param2 model_name corruption severity
  for entry in "${COMBINATIONS[@]}"; do
    IFS='|' read -r seed_idx model param1 param2 <<< "$entry"
    model_name="$(build_model_name "$model" "$param1" "$param2")"
    for corruption in "${CORRUPTIONS[@]}"; do
      for severity in "${SEVERITIES[@]}"; do
        expected_json_path "$corruption" "$severity" "$seed_idx" "$model_name"
      done
    done
  done
}

emit_missing_files() {
  local expected_path
  while IFS= read -r expected_path; do
    if [[ ! -s "$expected_path" ]]; then
      printf '%s\n' "$expected_path"
    fi
  done < <(emit_expected_files)
}

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

emit_missing_content_files() {
  local required_top_level_keys required_t_dict_links
  required_top_level_keys="$(join_by_comma "${REQUIRED_TOP_LEVEL_KEYS[@]}")"
  required_t_dict_links="$(join_by_comma "${REQUIRED_T_DICT_LINKS[@]}")"

  emit_expected_files | python -c '
import json
import sys
from pathlib import Path

required_top_level_keys = [key for key in sys.argv[1].split(",") if key]
required_t_dict_links = [key for key in sys.argv[2].split(",") if key]

for raw_path in sys.stdin:
    path_text = raw_path.strip()
    if not path_text:
        continue

    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        continue

    try:
        with path.open() as handle:
            data = json.load(handle)
    except Exception:
        print(path_text)
        continue

    if not isinstance(data, dict):
        print(path_text)
        continue

    if any(key not in data for key in required_top_level_keys):
        print(path_text)
        continue

    if required_t_dict_links:
        t_dict = data.get("T_dict")
        if not isinstance(t_dict, dict):
            print(path_text)
            continue

        if any(link not in t_dict for link in required_t_dict_links):
            print(path_text)
' "$required_top_level_keys" "$required_t_dict_links"
}

parse_rerun_target() {
  local target_path="$1"
  local target_abs eval_abs rel_path corrsev result_file result_stem

  target_abs="$(readlink -f "$target_path")"
  eval_abs="$(readlink -f "$EVAL_BASE")"
  rel_path="${target_abs#"$eval_abs"/}"

  if [[ "$rel_path" == "$target_abs" ]]; then
    rel_path="${target_path#"$EVAL_BASE"/}"
    rel_path="${rel_path#./}"
  fi

  corrsev="${rel_path%%/*}"
  rel_path="${rel_path#*/}"
  SEED_IDX="${rel_path%%/*}"
  result_file="${rel_path#*/}"

  if [[ -z "$corrsev" || -z "$SEED_IDX" || "$result_file" == "$rel_path" ]]; then
    echo "Could not parse rerun target: $target_path" >&2
    exit 1
  fi

  CORRUPTION="${corrsev%-*}"
  SEVERITY="${corrsev##*-}"
  result_stem="${result_file%.json}"
  result_stem="${result_stem%_"$EPOCH"}"
  MODEL_NAME="resnet50_${result_stem}"
}

run_eval() {
  local seed_idx="$1"
  local model_name="$2"
  local corruption="$3"
  local severity="$4"
  local save_path="$SAVE_BASE/${seed_idx}/"
  local save_eval_path="$EVAL_BASE/${corruption}-${severity}/${seed_idx}/"
  local model_file="${model_name}_${EPOCH}.model"

  mkdir -p "$save_eval_path"

  echo "Saved Models Path: $save_path"
  echo "Model Filename: $model_file"
  echo "Evaluating: SEED_IDX=$seed_idx, MODEL=$model_name"
  echo "Evaluating corruption=$corruption severity=$severity"
  echo "Save Eval Path: $save_eval_path"

  if [ "$DEBUG" = false ]; then
    python ../evaluate.py \
      --dataset cifar10_c \
      --dataset-root "$DATASET_ROOT" \
      --corruption "$corruption" \
      --severity "$severity" \
      --model resnet50 \
      -log \
      $GPU_FLAG \
      $SMOKE_ARG \
      --save-path "$save_path" \
      --save-eval-path "$save_eval_path" \
      --saved_model_name "$model_file" \
      --links "${EVALUATE_LINKS[@]}" \
      --seed "$seed_idx" \
      >> "${save_eval_path}/${model_name}.txt"
  fi
}

print_sbatch_command() {
  local mode="$1"
  local job_count="$2"

  if [[ "$job_count" -gt 0 ]]; then
    echo "Array jobs needed: $job_count"
    echo "Submit with:"
    echo "  sbatch --array=0-$((job_count - 1)) $SCRIPT_PATH $mode"
  else
    echo "Array jobs needed: 0"
  fi
}

print_help() {
  cat <<EOF
Usage:
  bash $SCRIPT_PATH help
  bash $SCRIPT_PATH manifest
  bash $SCRIPT_PATH missing-file
  bash $SCRIPT_PATH missing-content
  sbatch --array=0-<N-1> $SCRIPT_PATH selective-run-missing-file
  sbatch --array=0-<N-1> $SCRIPT_PATH selective-run-missing-content
  sbatch --array=0-<N-1> $SCRIPT_PATH run-from-scratch

Modes:
  help                          Print this message. This is the default mode.
  manifest                      Write the full list of expected JSON outputs.
  missing-file                  Write expected JSON outputs that are absent or empty.
  missing-content               Write existing JSON outputs that miss required content.
  selective-run-missing-file    Run one SLURM array task per missing file.
  selective-run-missing-content Run one SLURM array task per file with missing content.
  run-from-scratch              Run the original full evaluation schedule.

Configured paths:
  EVAL_BASE:                  $EVAL_BASE
  EXPECTED_FILES_LIST:        $EXPECTED_FILES_LIST
  MISSING_FILES_LIST:         $MISSING_FILES_LIST
  MISSING_CONTENT_FILES_LIST: $MISSING_CONTENT_FILES_LIST

Configured evaluation links:
  ${EVALUATE_LINKS[*]}

Required content for missing-content mode:
  Top-level keys: ${REQUIRED_TOP_LEVEL_KEYS[*]:-(none)}
  T_dict links:   ${REQUIRED_T_DICT_LINKS[*]:-(none)}

Current schedule:
  Scratch array jobs: $TOTAL_JOBS

Suggested workflow:
  1. Edit the configuration variables in this script.
  2. Run: bash $SCRIPT_PATH manifest
  3. Run: bash $SCRIPT_PATH missing-file
  4. Run: bash $SCRIPT_PATH missing-content
  5. Submit the selective-run command printed by the relevant missing mode.
EOF
}

case "$MODE" in
  help|-h|--help)
    print_help
    exit 0
    ;;
  manifest)
    mkdir -p "$(dirname "$EXPECTED_FILES_LIST")"
    emit_expected_files > "$EXPECTED_FILES_LIST"
    echo "Wrote expected file manifest: $EXPECTED_FILES_LIST"
    echo "Expected files: $(wc -l < "$EXPECTED_FILES_LIST")"
    print_sbatch_command "run-from-scratch" "$TOTAL_JOBS"
    exit 0
    ;;
  missing-file|missing)
    mkdir -p "$(dirname "$MISSING_FILES_LIST")"
    emit_missing_files > "$MISSING_FILES_LIST"
    MISSING_COUNT="$(wc -l < "$MISSING_FILES_LIST")"
    echo "Wrote missing file list: $MISSING_FILES_LIST"
    echo "Missing files: $MISSING_COUNT"
    print_sbatch_command "selective-run-missing-file" "$MISSING_COUNT"
    exit 0
    ;;
  missing-content|missing_content)
    mkdir -p "$(dirname "$MISSING_CONTENT_FILES_LIST")"
    emit_missing_content_files > "$MISSING_CONTENT_FILES_LIST"
    MISSING_CONTENT_COUNT="$(wc -l < "$MISSING_CONTENT_FILES_LIST")"
    echo "Wrote missing-content file list: $MISSING_CONTENT_FILES_LIST"
    echo "Files with missing content: $MISSING_CONTENT_COUNT"
    echo "Required top-level keys: ${REQUIRED_TOP_LEVEL_KEYS[*]:-(none)}"
    echo "Required T_dict links: ${REQUIRED_T_DICT_LINKS[*]:-(none)}"
    print_sbatch_command "selective-run-missing-content" "$MISSING_CONTENT_COUNT"
    exit 0
    ;;
  selective-run|selective_run|selective|selective-run-missing-file|selective_run_missing_file)
    if [[ ! -s "$MISSING_FILES_LIST" ]]; then
      echo "Missing file list is empty or does not exist: $MISSING_FILES_LIST" >&2
      echo "Run: bash $SCRIPT_PATH missing-file" >&2
      exit 1
    fi
    mapfile -t RERUN_TARGETS < "$MISSING_FILES_LIST"
    TOTAL_JOBS=${#RERUN_TARGETS[@]}
    ;;
  selective-run-missing-content|selective_run_missing_content)
    if [[ ! -s "$MISSING_CONTENT_FILES_LIST" ]]; then
      echo "Missing-content file list is empty or does not exist: $MISSING_CONTENT_FILES_LIST" >&2
      echo "Run: bash $SCRIPT_PATH missing-content" >&2
      exit 1
    fi
    mapfile -t RERUN_TARGETS < "$MISSING_CONTENT_FILES_LIST"
    TOTAL_JOBS=${#RERUN_TARGETS[@]}
    ;;
  run-from-scratch|run_from_scratch|run)
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Valid modes: help, manifest, missing-file, missing-content, selective-run-missing-file, selective-run-missing-content, run-from-scratch" >&2
    exit 1
    ;;
esac

# Dynamically determine SLURM_ARRAY_TASK_ID upper bound
if [ "$DEBUG" = true ]; then
  echo "Total jobs: $TOTAL_JOBS"
  if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
  fi
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  print_sbatch_command "$MODE" "$TOTAL_JOBS"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "Invalid array index: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

# Environment setup (modify if needed)
source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate focal_scaling

if [[ "$MODE" == "selective-run" || "$MODE" == "selective_run" || "$MODE" == "selective" \
   || "$MODE" == "selective-run-missing-file" || "$MODE" == "selective_run_missing_file" \
   || "$MODE" == "selective-run-missing-content" || "$MODE" == "selective_run_missing_content" ]]; then
  parse_rerun_target "${RERUN_TARGETS[$SLURM_ARRAY_TASK_ID]}"
  run_eval "$SEED_IDX" "$MODEL_NAME" "$CORRUPTION" "$SEVERITY"
  exit 0
fi

# Parse the selected combination
entry=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}
IFS='|' read -r SEED_IDX MODEL PARAM1 PARAM2 <<< "$entry"
MODEL_NAME="$(build_model_name "$MODEL" "$PARAM1" "$PARAM2")"

# Log info
echo "Evaluating: SEED_IDX=$SEED_IDX, MODEL=$MODEL_NAME"
if [ "$DEBUG" = true ]; then
  printf '%s\n' "${COMBINATIONS[@]}"
fi

# exit 0

# Run evaluation
for corruption in "${CORRUPTIONS[@]}"; do
  for severity in "${SEVERITIES[@]}"; do
    run_eval "$SEED_IDX" "$MODEL_NAME" "$corruption" "$severity"
  done
done
