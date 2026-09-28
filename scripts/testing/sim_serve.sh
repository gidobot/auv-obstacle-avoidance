#!/bin/bash
# Web viewer: planned mission underlay plus the live vehicle track.
#
#   ./sim_serve.sh [mission.xml] [--port 8095] [...]
#
# The mission may be a host path (anywhere under this repo or the acfr-lcm
# clone) or an in-container path; it is translated either way.  With no
# mission the viewer still runs, just without the planned underlay.
#
# Open http://localhost:<port>/ on the host -- the container is on host
# networking, so no port mapping is needed.
set -uo pipefail
. "$(dirname "$0")/sim_common.sh"

MISSION=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    MISSION=$(to_container_path "$1"); shift
    echo "mission: $MISSION"
fi
# pick the port out of the pass-through args purely so the hint below is right
PORT=8095
prev=""
for a in "$@"; do
    [ "$prev" = "--port" ] && PORT="$a"
    prev="$a"
done

SIM_CMD="python3 ${OA_DIR}/sitl_eval.py serve \
    --vehicle ${VEHICLE} \
    ${MISSION:+--mission ${MISSION}} \
    --lcmtypes \$(ls -d ${LCMTYPES_GLOB} 2>/dev/null | head -1) \
    $*"
echo "viewer will be at http://localhost:${PORT}/"
sim_exec
