#!/bin/bash
# Live mission progress in the terminal.  No GUI, so it works over ssh.
#
#   ./sim_watch.sh [--interval 2] [--regress 1] [...]
#
# Any extra arguments go through to `sitl_eval.py watch`.  Warns on the
# failure modes that are invisible in a summary: distance to the goal rising
# (a replan anchored behind the vehicle), PATH_RESPONSE stopping (the planner
# stack is down), and legs running long.
set -uo pipefail
. "$(dirname "$0")/sim_common.sh"

SIM_CMD="python3 ${OA_DIR}/sitl_eval.py watch \
    --vehicle ${VEHICLE} \
    --lcmtypes \$(ls -d ${LCMTYPES_GLOB} 2>/dev/null | head -1) \
    $*"
sim_exec
