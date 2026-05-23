#!/usr/bin/env bash
# Wait for T4b to drain, then run T5.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while pgrep -af "swift\.cli\.main sft" > /dev/null; do
    sleep 10
done
sleep 5

echo "=== chain: T5 launching $(date) ==="
bash "${SCRIPT_DIR}/launch_T5.sh"
echo "=== chain: T5 done (exit $?) $(date) ==="
