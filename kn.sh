#!/bin/bash
# Usage: kn.sh script.py [args...]
# Profile a script with nsys and print the GPU kernel names it launched.
# XLA runs cuDNN kernels inside CUDA graphs, hence --cuda-graph-trace=node.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=${PYTHONPATH:-$PWD/pylib} CUDA_VISIBLE_DEVICES=${GPU:-0}
rm -f kn.nsys-rep kn.sqlite
nsys profile -o kn --force-overwrite true -t cuda --cuda-graph-trace=node \
  python "$@" >/dev/null 2>&1
nsys export -t sqlite -o kn.sqlite --force-overwrite true kn.nsys-rep \
  >/dev/null 2>&1
python - <<'PY'
import sqlite3

con = sqlite3.connect("kn.sqlite")
rows = con.execute(
    "select s.value, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k "
    "join StringIds s on s.id = k.demangledName "
    "group by s.value order by 2 desc")
for name, count in rows:
  print(count, name[:220])
PY
