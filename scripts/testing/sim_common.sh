#!/bin/bash
# Shared setup for the SITL testing helpers.
#
# These wrap `docker exec` into the seeker-gazebo container, which is where
# sitl_eval.py has to run: it needs both the LCM stack and, for contacts, ROS.
# The container mounts this repo at /oa and the acfr-lcm clone at /acfr-lcm
# (see acfr_sim_ws/docker-compose.yaml), so recordings written to /oa/runs land
# straight on the host with no docker cp in either direction.
#
# Override any of these from the environment:
#   SIM_CONTAINER   container name        (default: discovered by name match)
#   VEHICLE         LCM vehicle name      (default: SEEKER-SITL)
#   OA_DIR          this repo inside the container   (default: /oa)
#   ACFR_LCM_DIR_C  acfr-lcm inside the container    (default: /acfr-lcm)
set -uo pipefail

VEHICLE="${VEHICLE:-SEEKER-SITL}"
OA_DIR="${OA_DIR:-/oa}"
ACFR_LCM_DIR_C="${ACFR_LCM_DIR_C:-/acfr-lcm}"

# The generated OA types (auv_oa_command_t) exist only in acfr-lcm's build
# tree; the acfr-lcm-types package in the image does not carry them.  Globbed
# because the tree is generated against whatever Python built it.
LCMTYPES_GLOB="${ACFR_LCM_DIR_C}/build/lib/python3.*/dist-packages/perls/lcmtypes"

find_container() {
    if [ -n "${SIM_CONTAINER:-}" ]; then
        echo "$SIM_CONTAINER"; return 0
    fi
    local c
    c=$(docker ps --format '{{.Names}}' | grep -i 'seeker-gazebo' | head -1)
    if [ -z "$c" ]; then
        echo "error: no running seeker-gazebo container." >&2
        echo "       start the sim first:  cd acfr_sim_ws && ./run_docker_dev.sh" >&2
        echo "       or set SIM_CONTAINER=<name>" >&2
        return 1
    fi
    echo "$c"
}

# Run a command inside the sim container with ROS sourced.
# sim_exec [extra docker-exec flags...]  -- runs $SIM_CMD inside the container.
#
# -it is added only when there is a terminal on both ends.  Interactively that
# is what makes ctrl-C reach the process inside the container; from a script or
# a pipe `docker exec -it` fails outright with "cannot attach stdin to a
# TTY-enabled container", so the flags have to be conditional rather than
# assumed.
sim_exec() {
    local c; c=$(find_container) || return 1
    local tty=()
    if [ -t 0 ] && [ -t 1 ]; then tty=(-it); fi
    docker exec "${tty[@]}" "$@" "$c" \
        bash -lc ". /opt/ros/jazzy/setup.sh >/dev/null 2>&1; $SIM_CMD"
}

# Resolve a host path to its in-container equivalent when it lives in a
# mounted tree, so the helpers can be given either form.
to_container_path() {
    local p="$1"
    case "$p" in
        /oa/*|/acfr-lcm/*|/tmp/*) echo "$p"; return ;;
    esac
    local abs; abs=$(readlink -f "$p" 2>/dev/null || echo "$p")
    local oa_host acfr_host
    oa_host=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null)
    acfr_host=$(readlink -f "${ACFR_LCM_HOST:-$oa_host/../acfr-lcm}" 2>/dev/null)
    case "$abs" in
        "$oa_host"/*)   echo "${OA_DIR}/${abs#$oa_host/}" ;;
        "$acfr_host"/*) echo "${ACFR_LCM_DIR_C}/${abs#$acfr_host/}" ;;
        *) echo "$abs" ;;
    esac
}
