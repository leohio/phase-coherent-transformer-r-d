#!/bin/bash
# Multi-GPU entrypoint for h100-8gpu-80gb DOK plan.
# Runs up to 8 training jobs in parallel, one per GPU (CUDA_VISIBLE_DEVICES).
#
# Each job's spec is in env vars JOB_<i>_<FIELD> for i in 0..7.
# Common architecture params (DIM, DEPTH, HEADS, ...) shared across jobs.
# Skip slot if JOB_<i>_ID is empty.
#
# Job output → ${SAKURA_ARTIFACT_DIR}/output/<JOB_ID>/{final_model.pt, metrics.jsonl, summary.json, log.txt}
# Persistent checkpoints → /workspace/ckpt/<JOB_ID>/

set -uo pipefail

if [ -z "${SAKURA_ARTIFACT_DIR:-}" ]; then
    SAKURA_ARTIFACT_DIR="/workspace/output"
fi
mkdir -p "${SAKURA_ARTIFACT_DIR}/output"

echo "================================================================"
echo " Phase 8 DOK 8GPU worker starting"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | head -10
echo "================================================================"

cd /workspace/phase8_src

PIDS=()
JOB_IDS=()
START_TS=$(date +%s)

for i in 0 1 2 3 4 5 6 7; do
    ID_VAR="JOB_${i}_ID"
    JOB_ID="${!ID_VAR:-}"
    if [ -z "${JOB_ID}" ]; then continue; fi

    TASK_VAR="JOB_${i}_TASK";          JOB_TASK="${!TASK_VAR:-}"
    PARAMS_VAR="JOB_${i}_TASK_PARAMS"; JOB_PARAMS="${!PARAMS_VAR:-{}}"
    CELL_VAR="JOB_${i}_CELL";          JOB_CELL="${!CELL_VAR:-}"
    SEED_VAR="JOB_${i}_SEED";          JOB_SEED="${!SEED_VAR:-0}"
    STEPS_VAR="JOB_${i}_STEPS";        JOB_STEPS="${!STEPS_VAR:-30000}"
    BS_VAR="JOB_${i}_BATCH";           JOB_BS="${!BS_VAR:-32}"
    LR_VAR="JOB_${i}_LR";              JOB_LR="${!LR_VAR:-3e-4}"
    WARMUP_VAR="JOB_${i}_WARMUP";      JOB_WARMUP="${!WARMUP_VAR:-1000}"
    CKPT_VAR="JOB_${i}_CHECKPOINT_EVERY"; JOB_CKPT="${!CKPT_VAR:-1000}"
    EVAL_VAR="JOB_${i}_EVAL_EVERY";    JOB_EVAL="${!EVAL_VAR:-500}"

    OUT_DIR="${SAKURA_ARTIFACT_DIR}/output/${JOB_ID}"
    CKPT_DIR="/workspace/ckpt/${JOB_ID}"
    mkdir -p "${OUT_DIR}" "${CKPT_DIR}"

    echo "[GPU ${i}] launching ${JOB_ID} (task=${JOB_TASK} cell=${JOB_CELL} seed=${JOB_SEED} batch=${JOB_BS})"

    CUDA_VISIBLE_DEVICES=${i} python3 -u train_with_checkpoint.py \
        --task "${JOB_TASK}" \
        --task-params "${JOB_PARAMS}" \
        --cell "${JOB_CELL}" \
        --seed "${JOB_SEED}" \
        --steps "${JOB_STEPS}" \
        --batch-size "${JOB_BS}" \
        --lr "${JOB_LR}" \
        --warmup "${JOB_WARMUP}" \
        --checkpoint-every "${JOB_CKPT}" \
        --eval-every "${JOB_EVAL}" \
        --dim "${DIM:-256}" \
        --depth "${DEPTH:-6}" \
        --heads "${HEADS:-8}" \
        --dim-head "${DIM_HEAD:-32}" \
        --ff-mult "${FF_MULT:-4}" \
        --output-dir "${OUT_DIR}" \
        --checkpoint-dir "${CKPT_DIR}" \
        --mgr-job-id "${JOB_ID}" \
        > "${OUT_DIR}/log.txt" 2>&1 &
    PIDS+=($!)
    JOB_IDS+=("${JOB_ID}")
done

echo "[8GPU] launched ${#PIDS[@]} jobs, waiting for completion..."

# Wait for all + collect exit codes
EXIT_CODES=()
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    jid="${JOB_IDS[$idx]}"
    wait "${pid}"
    rc=$?
    EXIT_CODES+=($rc)
    if [ $rc -eq 0 ]; then
        echo "[8GPU] job ${jid} completed (exit 0)"
    else
        echo "[8GPU] job ${jid} FAILED (exit ${rc})"
    fi
done

ELAPSED=$(($(date +%s) - START_TS))
echo "================================================================"
echo " 8GPU container done. ${#PIDS[@]} jobs in ${ELAPSED}s"
echo " Exit codes: ${EXIT_CODES[@]}"
echo "================================================================"

# Write batch summary for launcher to parse
{
    echo "{"
    echo "  \"container_jobs\": ${#PIDS[@]},"
    echo "  \"elapsed_s\": ${ELAPSED},"
    echo "  \"job_results\": ["
    for idx in "${!JOB_IDS[@]}"; do
        if [ $idx -gt 0 ]; then echo ","; fi
        echo -n "    {\"job_id\": \"${JOB_IDS[$idx]}\", \"exit_code\": ${EXIT_CODES[$idx]}}"
    done
    echo ""
    echo "  ]"
    echo "}"
} > "${SAKURA_ARTIFACT_DIR}/output/batch_summary.json"

# Container exits 0 if at least one job succeeded; non-zero only if ALL failed
ANY_OK=0
for rc in "${EXIT_CODES[@]}"; do
    if [ $rc -eq 0 ]; then ANY_OK=1; break; fi
done
[ $ANY_OK -eq 1 ] && exit 0 || exit 1
