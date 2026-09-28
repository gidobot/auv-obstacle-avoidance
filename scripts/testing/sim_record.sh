#!/bin/bash
# Record a run to a .npz for scoring.
#
#   ./sim_record.sh runs/new.npz --label cliff-manifold
#
# Records until interrupted (ctrl-C, or SIGTERM from outside), autosaving
# every 60 s so a hard kill does not lose the run.  The output path may be a
# host path under this repo -- it is translated to the mounted location, so
# the file lands on the host ready for `sitl_eval.py report`.
set -uo pipefail
. "$(dirname "$0")/sim_common.sh"

if [ $# -lt 1 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1
fi
OUT=$(to_container_path "$1"); shift
echo "writing: $OUT"

SIM_CMD="python3 ${OA_DIR}/sitl_eval.py record ${OUT} \
    --vehicle ${VEHICLE} \
    --lcmtypes \$(ls -d ${LCMTYPES_GLOB} 2>/dev/null | head -1) \
    $*"
sim_exec
