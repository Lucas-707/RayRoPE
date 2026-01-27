#! /bin/bash
source ~/.bashrc
conda activate rayrope

# Machine-dependent paths
HOSTNAME=$(hostname)
LOG_ROOT="./rayrope_out" # where model ckpts, logs, visual outputs are saved. Feel free to change.

# Dataset paths
# TODO: Replace the following paths with your dataset paths
export RE10K_TRAIN_DIR="/grogu/datasets/realestate/re10k_processed/train" # Replace with {YOUR_RE10K_DIR}/re10k_processed/train
export RE10K_TEST_DIR="/grogu/datasets/realestate/re10k_processed/test" # Replace with {YOUR_RE10K_DIR}/re10k_processed/test
export OBJV_DIR="/grogu/user/yuwu3/objaverse80k_sp/data" # Replace with the path to your Objaverse
export CO3D_DIR="/grogu/datasets/co3d"
export CO3D_ANNOTATION_DIR="/grogu/user/amylin2/co3d_v2_annotations"
export CO3D_DEPTH_DIR="/grogu/datasets/co3d"

# Auto-detect GPUs
# NGPUS=$(nvidia-smi -L | wc -l)
# GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')

# Or manually set GPUs
NGPUS=1
GPU_INDICES="0"
export CUDA_VISIBLE_DEVICES=$GPU_INDICES
echo "Using $NGPUS GPUs: $CUDA_VISIBLE_DEVICES"

export PYTHONPATH=${PYTHONPATH}:.
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}_${RANK}
export TORCHINDUCTOR_DISABLE_AUTOTUNE_CACHE=1
export TORCHINDUCTOR_FORCE_RECOMPILE=1

# Defaults
DATASET="co3d"
RAY_ENCODING="camray"
POS_ENC="prope"
ROPE_T_TYPE="none"
P0_TYPE="ex3d"
DEPTH_TYPE="none"
DENC_TYPE="inv_d"
INIT_D=0.0
INIT_SIG=3.0
INPUT_DEPTH="false"
FREQ_BASE=3.0
NUM_RAYS=3
DISABLE_VO="false"
SEED=1
MODEL_DIM=1152
BATCH_SIZE=4
NHEAD=8
NUM_LAYERS=6
DIM_FEEDFORWARD=1024
MAX_STEPS=80000
TEST_EVERY=80000
TEST=false
TEST_UNSEEN=false
CATEGORY="seen"
P_LOSS_W=0.5
BG_LOSS_W=1.0
PDB_MODE=false
TEST_N=200 # test first N scenes when rendering video/view is enabled


# Argument Parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset) DATASET="$2"; shift 2 ;;
    --ray_encoding) RAY_ENCODING="$2"; shift 2 ;;
    --pos_enc) POS_ENC="$2"; shift 2 ;;
    --rope_t_type) ROPE_T_TYPE="$2"; shift 2 ;;
    --p0_type) P0_TYPE="$2"; shift 2 ;;
    --depth_type) DEPTH_TYPE="$2"; shift 2 ;;
    --denc_type) DENC_TYPE="$2"; shift 2 ;;
    --init_d) INIT_D="$2"; shift 2 ;;
    --init_sig) INIT_SIG="$2"; shift 2 ;;
    --input_depth) INPUT_DEPTH="true"; shift 1 ;;
    --freq_base) FREQ_BASE="$2"; shift 2 ;;
    --num_rays) NUM_RAYS="$2"; shift 2 ;;
    --disable_vo) DISABLE_VO="true"; shift 1 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model_dim) MODEL_DIM="$2"; shift 2 ;;
    --batch) BATCH_SIZE="$2"; shift 2 ;;
    --nhead) NHEAD="$2"; shift 2 ;;
    --num_layers) NUM_LAYERS="$2"; shift 2 ;;
    --dim_feedforward) DIM_FEEDFORWARD="$2"; shift 2 ;;
    --p_loss_w) P_LOSS_W="$2"; shift 2 ;;
    --bg_loss_w) BG_LOSS_W="$2"; shift 2 ;;
    --test) TEST=true; shift 1 ;;
    --test-render-video) TEST_RENDER_VIDEO=true; shift 1 ;;
    --test-rad-sph) TEST_RAD_SPH=true; shift 1 ;;
    --test-unseen) TEST_UNSEEN=true; shift 1 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --test-context-views) TEST_CONTEXT_VIEWS="$2"; shift 2 ;;
    --test-zoom-in) TEST_ZOOM_IN="$2"; shift 2 ;;
    --test-ckpt) TEST_CKPT="$2"; shift 2 ;;
    --pdb) PDB_MODE=true; shift 1 ;;
    -h|--help) echo "Usage: $0 --dataset <re10k|objaverse|co3d> [options]"; exit 0 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done



# Logs & Name
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NGPUS))
MODEL_CONFIG="L${NUM_LAYERS}-H${NHEAD}-D${MODEL_DIM}-FF${DIM_FEEDFORWARD}-B${EFFECTIVE_BATCH_SIZE}"

NAME="${POS_ENC}"
[[ "$P0_TYPE" != "ex3d" ]] && NAME="${NAME}-p0${P0_TYPE}"
[[ $NUM_RAYS != 3 ]] && NAME="${NAME}-${NUM_RAYS}ray"
[[ "$DEPTH_TYPE" != "none" ]] && NAME="${NAME}-${DEPTH_TYPE}"
[[ "$INIT_D" != "0.0" ]] && NAME="${NAME}-initd${INIT_D}"
[[ "$INIT_SIG" != "3.0" ]] && NAME="${NAME}-inits${INIT_SIG}"
[[ "$POS_ENC" == *"+"* ]] && NAME="${NAME}-${DENC_TYPE}"
[[ "$INPUT_DEPTH" == "true" ]] && NAME="${NAME}-indepth"
[[ $FREQ_BASE != 3.0 ]] && NAME="${NAME}-fb${FREQ_BASE}"
[[ "$DISABLE_VO" == "true" ]] && NAME="${NAME}-no_vo"
[[ "$ROPE_T_TYPE" != "none" ]] && NAME="${NAME}-${ROPE_T_TYPE}"
[[ "$RAY_ENCODING" != "camray" ]] && NAME="${NAME}-${RAY_ENCODING}"
[[ $P_LOSS_W != 0.5 ]] && NAME="${NAME}-pw${P_LOSS_W}"
[[ $BG_LOSS_W != 1.0 ]] && NAME="${NAME}-bgw${BG_LOSS_W}"

NAME="${NAME}-seed${SEED}"

INPUT_DEPTH_STR=$([ "$INPUT_DEPTH" == "true" ] && echo "known_d" || echo "unknown_d")
DATASET_STR=$([ "$DATASET" == "co3d" ] && echo "${DATASET}_${CATEGORY}" || echo "${DATASET}")

LOG_DIR="${LOG_ROOT}/${MODEL_CONFIG}/${DATASET_STR}/${INPUT_DEPTH_STR}"
PRINT_LOG_DIR="./logs/${MODEL_CONFIG}/${DATASET_STR}/${INPUT_DEPTH_STR}"
[[ $BG_LOSS_W != 1.0 || $P_LOSS_W != 0.5 ]] && LOG_DIR="${LOG_DIR}/masked" && PRINT_LOG_DIR="${PRINT_LOG_DIR}/masked"

LOG_DIR="${LOG_DIR}/${NAME}"

mkdir -p "${LOG_DIR}"
mkdir -p "${PRINT_LOG_DIR}"
PRINT_LOG_FILE="${PRINT_LOG_DIR}/${NAME}.log"
echo "Log file: ${PRINT_LOG_FILE}"

# After setting path names, overwrite batch size to 1 if render video
if [ "$TEST_RENDER_VIDEO" = true ]; then
  BATCH_SIZE=1
  echo "Setting batch size to 1 for rendering."
fi

if [ "$PDB_MODE" = true ]; then
  REDIRECT=""
else
  REDIRECT=">> \"${PRINT_LOG_FILE}\" 2>&1"
fi

# Command
PYTHON_CMD=$([ "$PDB_MODE" = true ] && echo "python -m pdb" || echo "python")
BASE_CMD=("NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 ${PYTHON_CMD} -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$NGPUS")
BASE_CMD+=(
  "nvs/trainval.py lvsm"
  "--amp --amp_dtype fp16"
  "--dataset ${DATASET}"
  "--co3d_train_categories ${CATEGORY}"
  "--dataset_batch_scenes ${BATCH_SIZE}"
  "--dataset_supervise_views 1"
  "--perceptual_loss_w ${P_LOSS_W}"
  "--bg_loss_w ${BG_LOSS_W}"
  "--model_config.encoder.num_layers ${NUM_LAYERS}"
  "--model_config.encoder.layer.d_model ${MODEL_DIM}"
  "--model_config.encoder.layer.nhead ${NHEAD}"
  "--model_config.encoder.layer.dim_feedforward ${DIM_FEEDFORWARD}"
  "--model_config.encoder.layer.qk_norm"
  "--max_steps ${MAX_STEPS} --test_every ${TEST_EVERY}"
  "--model_config.ray_encoding ${RAY_ENCODING}"
  "--model_config.pos_enc ${POS_ENC}"
  "--model_config.p0_type ${P0_TYPE}"
  "--model_config.depth_type ${DEPTH_TYPE}"
  "--model_config.cam_transform_type none"
  "--model_config.rope_transform_type ${ROPE_T_TYPE}"
  "--model_config.freq_base ${FREQ_BASE}"
  "--model_config.num_rays_per_patch ${NUM_RAYS}"
  "--seed ${SEED}"
  "--output_dir ${LOG_DIR}"
)

[[ "$INPUT_DEPTH" == "true" ]] && BASE_CMD+=("--model_config.depth_input")
[[ "$DISABLE_VO" == "true" ]] && BASE_CMD+=("--model_config.disable_vo")
[[ "$TEST_UNSEEN" == "true" && "$DATASET" == "co3d" ]] && BASE_CMD+=("--co3d_test_unseen")
[[ "$BG_LOSS_W" != "1.0" ]] && BASE_CMD+=("--get_mask")
``
# Execution
if [ -n "$TEST_ZOOM_IN" ]; then
  for zoom in $TEST_ZOOM_IN; do
    echo "Testing zoom ${zoom}..."
    CMD=("${BASE_CMD[@]}" "--test_only --auto_resume" "--test_zoom_factor ${zoom}" "--test_subdir eval-zoom${zoom}x")
    eval "${CMD[@]} $REDIRECT"
  done
elif [ -n "$TEST_CONTEXT_VIEWS" ]; then
  for cv in $TEST_CONTEXT_VIEWS; do
    echo "Testing context ${cv}..."
    CMD=("${BASE_CMD[@]}" "--test_only --auto_resume" "--model_config.ref_views ${cv}" "--test_input_views ${cv}" "--test_subdir eval-context${cv}")
    if [ "$DATASET" == "co3d" ]; then
      CMD+=("--co3d_test_seen_index_file assets/co3d_test_context${cv}_seen.json")
      [[ "$TEST_UNSEEN" == "true" ]] && CMD+=("--co3d_test_unseen_index_file assets/co3d_test_context${cv}_unseen.json")
    elif [ "$DATASET" == "objaverse" ]; then
      CMD+=("--objaverse_test_index_file assets/objaverse_index_test_context${cv}_all.json")
    elif [ "$DATASET" == "re10k" ]; then
      CMD+=("--test_index_fp evaluation_index_re10k_context${cv}.json")
    fi

    if [ "$TEST_RENDER_VIDEO" == "true" ]; then
      CMD+=("--render_video" "--test_n ${TEST_N}")
    elif [ "$TEST_RENDER_VIEW" == "true" ]; then
      CMD+=("--render_view" "--test_n ${TEST_N}")
    fi
    eval "${CMD[@]} $REDIRECT"
  done
elif [ "$TEST_RENDER_VIDEO" == "true" ]; then
  echo "Rendering video..."
  CMD=("${BASE_CMD[@]}" "--test_only --auto_resume --render_video --test_n ${TEST_N}")
  eval "${CMD[@]} $REDIRECT"
elif [ "$TEST_RENDER_VIEW" == "true" ]; then
  echo "Rendering view..."
  CMD=("${BASE_CMD[@]}" "--test_only --auto_resume --render_view --test_n ${TEST_N}")
  eval "${CMD[@]} $REDIRECT"
elif [ "$TEST_RAD_SPH" == "true" ]; then
  echo "Testing on radial and spherical splits..."
  CMD=("${BASE_CMD[@]}" "--test_only --auto_resume --test_rad_sph")
  eval "${CMD[@]} $REDIRECT"
elif [ -n "$TEST_CKPT" ]; then
  echo "Testing ckpt: ${TEST_CKPT}..."
  CMD=("${BASE_CMD[@]}" "--test_only --auto_resume" "--overwrite_ckpt_dir ${TEST_CKPT}")
  eval "${CMD[@]} $REDIRECT"
elif [ "$TEST" == "true" ]; then
  echo "Testing..."
  CMD=("${BASE_CMD[@]}" "--test_only --auto_resume")
  eval "${CMD[@]} $REDIRECT"
else
  echo "Training..."
  CMD=("${BASE_CMD[@]}")
  eval "${CMD[@]} $REDIRECT"
fi