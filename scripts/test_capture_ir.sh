#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAPTURE_JSON="${REPO_DIR}/config/capture.json"
SERVER_BIN="${REPO_DIR}/coalsack/build/graph_proc_server"
VIEWER_BIN="${REPO_DIR}/build/stargazer_viewer"
BACKUP_JSON="${CAPTURE_JSON}.bak"

# Verify required binaries exist
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "ERROR: ${SERVER_BIN} not found. Build it first:" >&2
    echo "  cd ${REPO_DIR}/coalsack/build && make graph_proc_server" >&2
    exit 1
fi
if [[ ! -x "${VIEWER_BIN}" ]]; then
    echo "ERROR: ${VIEWER_BIN} not found." >&2
    exit 1
fi

# Switch pipeline to capture_ir (backup first)
cp "${CAPTURE_JSON}" "${BACKUP_JSON}"

cleanup() {
    # Restore capture.json
    if [[ -f "${BACKUP_JSON}" ]]; then
        cp "${BACKUP_JSON}" "${CAPTURE_JSON}"
        rm -f "${BACKUP_JSON}"
        echo "Restored ${CAPTURE_JSON}"
    fi
    # Kill all child processes of this script
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

python3 -c "
import json, sys
path = sys.argv[1]
d = json.load(open(path))
d['pipeline'] = 'capture_ir'
json.dump(d, open(path, 'w'), indent=2)
" "${CAPTURE_JSON}"
echo "Set pipeline to capture_ir"

# Enter isolated network namespace (no root, no host impact)
# NOTE: do NOT use exec here — we need the outer shell's EXIT trap to run on exit
# Use heredoc to avoid quoting hell with nested bash -c
unshare --user --net --map-root-user bash <<NSEOF
set -euo pipefail

# Set up loopback + dummy interface for broadcast
ip link set lo up
for i in \$(seq 1 24); do
    ip addr add 192.168.0.\${i}/32 dev lo
done
# lo lacks BROADCAST flag; use dummy interface for broadcast
ip link add dev eth0 type dummy
ip link set eth0 up
ip addr add 192.168.0.254/24 broadcast 192.168.0.255 dev eth0
# broadcast_talker sends to 255.255.255.255 (limited broadcast); needs a default route
ip route add default dev eth0
echo 'Network interfaces ready'
ip addr show lo | grep 'inet '
ip addr show eth0 | grep 'inet '

SERVER_LOG=/tmp/stargazer_server.log

# Start graph_proc_server
${SERVER_BIN} > "\${SERVER_LOG}" 2>&1 &
SERVER_PID=\$!
echo "graph_proc_server started (pid=\${SERVER_PID}, log=\${SERVER_LOG})"
sleep 1

# Verify server is listening
if ! ss -tlnp 2>/dev/null | grep -q ':31400'; then
    echo 'WARNING: server does not appear to be listening on 31400' >&2
fi

# Launch viewer (foreground; server runs in background)
echo 'Starting stargazer_viewer ...'
cd ${REPO_DIR}/build
./stargazer_viewer || true

# Viewer exited — stop the server
echo 'Viewer closed. Stopping graph_proc_server ...'
kill \${SERVER_PID} 2>/dev/null || true
wait \${SERVER_PID} 2>/dev/null || true
NSEOF
