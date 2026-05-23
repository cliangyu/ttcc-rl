#!/usr/bin/env bash
# Wait for T1b to finish, then run T1c, then T3a. Each writes its own log.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Wait for any in-progress swift sft to drain
while pgrep -af "swift\.cli\.main sft" > /dev/null; do
    sleep 5
done
sleep 5  # let GPUs fully release

echo "=== chain: T1c launching $(date) ==="
bash "${SCRIPT_DIR}/launch_T1c.sh"
T1C_EXIT=$?
echo "=== chain: T1c done (exit ${T1C_EXIT}) $(date) ==="

# Drain again
while pgrep -af "swift\.cli\.main sft" > /dev/null; do
    sleep 5
done
sleep 5

echo "=== chain: T3a launching $(date) ==="
bash "${SCRIPT_DIR}/launch_T3a.sh"
T3A_EXIT=$?
echo "=== chain: T3a done (exit ${T3A_EXIT}) $(date) ==="

echo "=== chain complete ==="
