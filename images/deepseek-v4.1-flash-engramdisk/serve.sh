#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 on 4x RTX PRO 6000 (TP4, 512K): the exact container measured in
# targets/deepseek-v4.1-flash-4x-rtx-pro-6000.md and submissions/4x-rtx-pro-6000-96gb/2026-10-21-0xSero.json.
#
#   MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017 CACHE_DIR=/srv/cache/ds41 images/deepseek-v4.1-flash-engramdisk/serve.sh
#
# MODEL_DIR  the checkpoint at revision 1bae0e1088993b9b82de08f59de5be5a34e42170 (on local NVMe: Engram rows are read
#            from it at every decode step)
# CACHE_DIR  FlashInfer autotune / DeepGEMM JIT cache and the KV disk tier; the first start fills it (~22 min, use
#            GPU_MEM_UTIL=0.92 that once), later starts take ~7 min
# KV_OFFLOAD_GIB  pinned host KV region in /dev/shm (default 32; must fit the free /dev/shm, tmpfs = half of RAM)
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the DeepSeek-V4.1-Flash-1017 checkpoint}"
: "${CACHE_DIR:?set CACHE_DIR to a writable cache directory}"
IMAGE=${IMAGE:-ghcr.io/sybil-solutions/vllm-moet-dsv41-engramdisk@sha256:f3d7a175e8ccb89507ca66974386e724734d0e7ed0080d21d5518daf88aca6bd}
NAME=${NAME:-ds41-flash}
PORT=${PORT:-8000}
BIND=${BIND:-127.0.0.1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.94}
KV_OFFLOAD_GIB=${KV_OFFLOAD_GIB:-32}

# A stopped server leaves its root-owned offload region behind; with it there the next start fails
# "Insufficient space in /dev/shm".
if ls /dev/shm/vllm_offload_*.mmap >/dev/null 2>&1; then
  echo "stale KV offload region(s) in /dev/shm (delete them first, e.g. sudo rm /dev/shm/vllm_offload_*.mmap):" >&2
  ls -la /dev/shm/vllm_offload_*.mmap >&2
  exit 1
fi
free_gib=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
if (( free_gib <= KV_OFFLOAD_GIB )); then
  echo "/dev/shm has ${free_gib} GiB free, KV_OFFLOAD_GIB=${KV_OFFLOAD_GIB} does not fit" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"/{dot-cache,deep_gemm,kvcache}

docker run -d --name "$NAME" --gpus all --ipc host -p "$BIND:$PORT:$PORT" \
  -v "$MODEL_DIR:/models:ro" \
  -v "$CACHE_DIR/dot-cache:/root/.cache:rw" \
  -v "$CACHE_DIR/deep_gemm:/root/.deep_gemm:rw" \
  -v "$CACHE_DIR/kvcache:/srv/kvcache:rw" \
  -e VLLM_MOET_B12X_MOE_FUSED=1 -e VLLM_MOET_SPARSE_MLA_LDG=1 -e VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 \
  -e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=16 \
  -e VLLM_MOET_ENGRAM_DISK=1 -e VLLM_MOET_ENGRAM_DISK_THREADS=16 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" \
  /models --served-model-name deepseek-v4.1-flash-1017 --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 4 --max-model-len 524288 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs 8 \
  --kv-cache-dtype nvfp4_ds_mla \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_tp_shard_min_rows":2048}' \
  --moe-backend b12x \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false,"moe_backend":"b12x"}' \
  --max-num-batched-tokens 4096 --max-cudagraph-capture-size 64 \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --kv-offloading-size "$KV_OFFLOAD_GIB" \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":"/srv/kvcache/v41-interleaved","n_read_threads":16,"n_write_threads":16,"locality":"LOCAL"}],"offload_prompt_only":false}}' \
  --max-logprobs 128

echo "started $NAME; ready when: curl -s http://$BIND:$PORT/v1/models | grep deepseek-v4.1-flash-1017"
