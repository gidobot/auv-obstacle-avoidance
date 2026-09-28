#!/bin/bash
# One-shot health check before trusting a run.
#
#   ./sim_status.sh
#
# Answers the questions that have silently invalidated runs in the past: is
# Gazebo actually alive, is the planner stack up, is ground truth being
# published, and is anything publishing ACFR_NAV twice.
set -uo pipefail
. "$(dirname "$0")/sim_common.sh"

SIM_CMD='
echo "=== sim processes ==="
printf "  gz sim:        %s\n" "$(ps aux | grep -c "[g]z sim")"
printf "  bridge:        %s\n" "$(ps aux | grep -c "[s]eeker_gazebo_bridge")"
echo "=== LCM channels (10s) ==="
python3 - <<PY
import time
from collections import Counter
import lcm
seen=Counter()
lc=lcm.LCM(); lc.subscribe(".*", lambda ch,d: seen.update([ch]))
t0=time.time()
while time.time()-t0 < 10: lc.handle_timeout(200)
V="'"${VEHICLE}"'"
def rate(c): return seen.get(c,0)/10.0
checks = [
 (f"{V}_GT.ACFR_NAV",   "ground truth", 5, 15),
 (f"{V}.ACFR_NAV",      "nav filter",    5, 15),
 (f"{V}.PATH_RESPONSE", "local planner", 1, 99),
 (f"{V}.OA_COMMAND",    "oa-mapper",     1, 99),
 (f"{V}.NUCLEUS.AHRS",  "sim AHRS",      1, 99),
]
for ch,label,lo,hi in checks:
    r=rate(ch)
    bad = "DOWN" if r<lo else ("DOUBLE PUBLISHER?" if r>hi else "ok")
    print(f"  {label:<14} {ch:<30} {r:6.2f} Hz  {bad}")
print(f"  total distinct channels: {len(seen)}")
PY'
sim_exec
