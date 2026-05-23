#!/usr/bin/env bash
# Polls ttcc_grpo_extended for new checkpoints and copies them to a mirror
# directory before save_total_limit deletes them.
CKPT_BASE=/home/ssm-user/work/work-out/ttcc_grpo_extended/v0-20260520-223244
MIRROR=/home/ssm-user/work/work-out/grpo_extended_ckpt_mirror
mkdir -p "$MIRROR"
while true; do
  for c in "$CKPT_BASE"/checkpoint-*; do
    [ -d "$c" ] || continue
    name=$(basename "$c")
    if [ ! -d "$MIRROR/$name" ]; then
      cp -r "$c" "$MIRROR/$name" && echo "[mirror] $name → $(date +%H:%M:%S)"
    fi
  done
  if ! pgrep -f "swift.cli.main rlhf" > /dev/null; then
    echo "[mirror] training process gone — exiting"
    break
  fi
  sleep 30
done
