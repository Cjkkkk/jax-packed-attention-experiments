#!/bin/bash
# Usage: [GPU=n] run.sh [packed_attn_repro.py args...]
cd "$(dirname "$0")"
export PYTHONPATH=${PYTHONPATH:-$PWD/pylib} CUDA_VISIBLE_DEVICES=${GPU:-0}
exec python packed_attn_repro.py "$@"
