#!/bin/bash
# DOK container entrypoint.
# Reads job spec from environment variables (set by orchestrator via DOK API)
# or CLI args if provided.
#
# Required env vars (set by orchestrator):
#   JOB_ID         : unique job identifier
#   TASK           : copymem | fftmnist | phase_memory | niah | radioml
#   TASK_PARAMS    : JSON dict (task-specific)
#   CELL           : real_softmax | ... | complex_screen
#   SEED           : integer
#   STEPS          : training steps
#   DIM, DEPTH, HEADS, DIM_HEAD, FF_MULT
#   LR, BATCH_SIZE
#   CHECKPOINT_EVERY, EVAL_EVERY
#   MGR_URL        : management server URL (heartbeats)
#
# Sakura DOK provides:
#   SAKURA_ARTIFACT_DIR : output dir auto-uploaded to artifacts on task end

set -euo pipefail

if [ -z "${SAKURA_ARTIFACT_DIR:-}" ]; then
    SAKURA_ARTIFACT_DIR="/workspace/output"
    mkdir -p "$SAKURA_ARTIFACT_DIR"
fi

OUTPUT_DIR="${SAKURA_ARTIFACT_DIR}/output"
CHECKPOINT_DIR="/workspace/ckpt"
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# Capture all entrypoint output (stdout+stderr) to artifact for post-mortem.
# DOK auto-uploads $SAKURA_ARTIFACT_DIR after container exits.
exec > >(tee -a "$SAKURA_ARTIFACT_DIR/console.log") 2>&1
echo "[entrypoint] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
trap 'rc=$?; echo "[entrypoint] exit rc=$rc at $(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit $rc' EXIT

# Default TASK_PARAMS to "{}" (avoid bash brace expansion bug with ${VAR:-{}})
if [ -z "${TASK_PARAMS:-}" ]; then
    TASK_PARAMS="{}"
fi

echo "================================================================"
echo " Phase 8 DOK worker starting"
echo " JOB_ID=${JOB_ID:-?}  TASK=${TASK:-?}  CELL=${CELL:-?}  SEED=${SEED:-?}"
echo " TASK_PARAMS=${TASK_PARAMS}"
echo " ARTIFACT_DIR=${SAKURA_ARTIFACT_DIR}"
python3 -c 'import torch; print(" GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")'
echo "================================================================"

cd /workspace/phase8_src

exec python3 train_with_checkpoint.py \
    --task "${TASK}" \
    --task-params "${TASK_PARAMS}" \
    --cell "${CELL}" \
    --seed "${SEED}" \
    --steps "${STEPS:-30000}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --lr "${LR:-3e-4}" \
    --warmup "${WARMUP:-1000}" \
    --dim "${DIM:-256}" \
    --depth "${DEPTH:-6}" \
    --heads "${HEADS:-8}" \
    --dim-head "${DIM_HEAD:-32}" \
    --ff-mult "${FF_MULT:-4}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-1000}" \
    --eval-every "${EVAL_EVERY:-500}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    ${MICRO_BATCH:+--micro-batch "${MICRO_BATCH}"} \
    ${ATTN_CHUNK_SIZE:+--attn-chunk-size "${ATTN_CHUNK_SIZE}"} \
    ${ATTN_GRAD_CKPT:+--attn-grad-checkpoint} \
    ${WEIGHT_DECAY:+--weight-decay "${WEIGHT_DECAY}"} \
    ${CLIP_GRAD:+--clip-grad "${CLIP_GRAD}"} \
    ${MGR_URL:+--mgr-server-url "${MGR_URL}"} \
    ${JOB_ID:+--mgr-job-id "${JOB_ID}"} \
    ${NO_SOFTMASK:+--no-softmask} \
    ${SOFTMASK:+--softmask}
