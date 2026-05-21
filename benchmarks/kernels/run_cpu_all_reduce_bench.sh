#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch benchmark_cpu_all_reduce.py with one rank per NUMA/SNC node, each
# pinned via numactl --cpunodebind + --membind for memory-local SHM writes.
#
# Usage:
#   bash benchmarks/kernels/run_cpu_all_reduce_bench.sh [script args...]
#
# Examples:
#   bash benchmarks/kernels/run_cpu_all_reduce_bench.sh --backend shm
#   bash benchmarks/kernels/run_cpu_all_reduce_bench.sh --backend gloo \
#       --output bench_gloo.json
#
# Environment overrides:
#   NODES       Space-separated NUMA node IDs to use (default: all from numactl)
#               Example: NODES="0 2" bash run_cpu_all_reduce_bench.sh
#   MASTER_ADDR Distributed rendezvous address (default: 127.0.0.1)
#   MASTER_PORT Distributed rendezvous port (default: 29500)
#
# On SNC-enabled Intel parts, each SNC cluster is already a separate NUMA node,
# so no special handling is needed — the auto-detection works unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH="${SCRIPT_DIR}/benchmark_cpu_all_reduce.py"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python not found at ${PYTHON}. Run 'uv venv' and install deps first." >&2
    exit 1
fi

if ! command -v numactl &> /dev/null; then
    echo "ERROR: numactl not found. Install it (e.g. apt-get install numactl)." >&2
    exit 1
fi

# Auto-detect NUMA nodes unless overridden.
if [[ -n "${NODES:-}" ]]; then
    read -ra NODES_ARRAY <<< "${NODES}"
else
    mapfile -t NODES_ARRAY < <(
        numactl --hardware \
            | awk '/^available:/ {for (i=3; i<=NF; i++) if ($i ~ /^[0-9]+$/) print $i}'
    )
fi

WORLD_SIZE=${#NODES_ARRAY[@]}
if [[ "${WORLD_SIZE}" -lt 2 ]]; then
    echo "ERROR: Need at least 2 NUMA nodes. Found: ${WORLD_SIZE}. " \
         "Override with NODES='0 1' if you have a single-socket machine." >&2
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_PORT
# Unique identifier so that multiple concurrent runs do not share SHM names.
export VLLM_DIST_IDENT="cpuar-$$-${MASTER_PORT}"

echo "NUMA nodes: ${NODES_ARRAY[*]}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "MASTER_ADDR: ${MASTER_ADDR}:${MASTER_PORT}"
echo "VLLM_DIST_IDENT: ${VLLM_DIST_IDENT}"
echo ""

pids=()
for rank in "${!NODES_ARRAY[@]}"; do
    node="${NODES_ARRAY[$rank]}"
    RANK="${rank}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    LOCAL_RANK="${rank}" \
        numactl --cpunodebind="${node}" --membind="${node}" \
        "${PYTHON}" -u "${BENCH}" "$@" &
    pids+=($!)
done

# Wait for all ranks; propagate non-zero exit codes.
exit_code=0
for p in "${pids[@]}"; do
    wait "${p}" || exit_code=$?
done
exit "${exit_code}"
