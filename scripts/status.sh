#!/usr/bin/env bash
# Quick state dump — what stage is the overnight pipeline in?
echo "=========================================================="
echo " TTCC overnight pipeline status  ($(date))"
echo "=========================================================="
echo
echo "## CoT distillation"
for shard in 0 1; do
    n=$(sudo cat "/home/ssm-user/work/work-out/cot/full_instruct_gpu${shard}.jsonl" 2>/dev/null | wc -l || echo 0)
    echo "  GPU${shard}: ${n} ads"
done
N0=$(sudo cat /home/ssm-user/work/work-out/cot/full_instruct_gpu0.jsonl 2>/dev/null | wc -l || echo 0)
N1=$(sudo cat /home/ssm-user/work/work-out/cot/full_instruct_gpu1.jsonl 2>/dev/null | wc -l || echo 0)
echo "  total: $((N0+N1)) / ~717 train ads"
echo

echo "## Live processes (ssm-user)"
sudo ps -ef | grep -E "cot_distill|run_all|swift" | grep ssm-user | grep -v grep | head -10
echo

echo "## GPU status"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv
echo

echo "## Output files in work-out/"
sudo ls -la /home/ssm-user/work/work-out/ 2>/dev/null | tail -30
echo
sudo ls -la /home/ssm-user/work/work-out/cot/ 2>/dev/null | tail
echo

echo "## run_all.log tail (orchestrator)"
sudo tail -15 /home/ssm-user/work/work-out/run_all.log 2>/dev/null
echo

echo "## Most recent run logs"
for log in /home/ssm-user/work/work-out/cot/full_gpu0.log \
           /home/ssm-user/work/work-out/cot/full_gpu1.log \
           /home/ssm-user/work/work-out/ttcc_sft/sft.log \
           /home/ssm-user/work/work-out/ttcc_grpo/grpo.log \
           /home/ssm-user/work/work-out/ttcc_rloo/rloo.log \
           /home/ssm-user/work/work-out/ttcc_gspo/gspo.log; do
    if sudo test -f "$log"; then
        echo "--- $log (last 5) ---"
        sudo tail -5 "$log" 2>/dev/null
    fi
done
echo
echo "## Disk"
df -h / /opt/dlami/nvme | tail
