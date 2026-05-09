# Phase 8 training container for Sakura DOK
# Base: PyTorch 2.4 with CUDA 12.4 (compatible with H100)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for science stack and SSH (for checkpoint sync to mgr VPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        openssh-client rsync \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps
COPY phase8_experiment/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy training code (existing complex_nn_experiment + phase8_experiment)
# These are baked into the image. Different jobs select via CLI args.
COPY complex_nn_experiment/ /workspace/complex_nn_experiment/
COPY phase8_experiment/src/ /workspace/phase8_src/
COPY phase8_experiment/scripts/entrypoint.sh /workspace/entrypoint.sh
COPY phase8_experiment/scripts/entrypoint_8gpu.sh /workspace/entrypoint_8gpu.sh

RUN chmod +x /workspace/entrypoint.sh /workspace/entrypoint_8gpu.sh

# Datasets baked into image for tasks needing non-synthetic data.
# Build time: copy contents of docker/datasets/ into /workspace/datasets/.
# For Path-X: place pathx.pt under docker/datasets/ before `docker build`.
# For tasks that don't need any baked dataset, leave docker/datasets/ empty.
RUN mkdir -p /workspace/datasets
COPY phase8_experiment/docker/datasets/ /workspace/datasets/
ENV PATHX_PATH=/workspace/datasets/pathx.pt \
    LRA_PATHFINDER_PATH=/workspace/datasets/lra_pathfinder.pt \
    LRA_TEXT_PATH=/workspace/datasets/lra_text.pt \
    LRA_LISTOPS_PATH=/workspace/datasets/lra_listops.pt

# Persistent checkpoint dir (DOK volume mount target)
RUN mkdir -p /workspace/ckpt

# DOK auto-uploads everything in $SAKURA_ARTIFACT_DIR after task completion.
# We'll write final_model.pt, metrics.jsonl, summary.json there.

ENTRYPOINT ["/workspace/entrypoint.sh"]
