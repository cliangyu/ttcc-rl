#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
python <<'PY'
from huggingface_hub import HfApi, login
from pathlib import Path

# Token: ssm-user already has it cached at ~/.cache/huggingface/token
token_path = Path("/home/ssm-user/.cache/huggingface/token")
if token_path.exists():
    login(token=token_path.read_text().strip(), add_to_git_credential=False)

api = HfApi()
repo_id = "liangyuch/ttcc-cot"
src = "/home/ssm-user/work/work-out/cot/v3_pro_merged.jsonl"
# Replace in place — same filename
dst = "cot_v2_causal_instruct.jsonl"

commit = api.upload_file(
    path_or_fileobj=src,
    path_in_repo=dst,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Replace CoT data: teacher = gemini-3.1-pro-preview (was Qwen3-Omni-30B-A3B-Instruct). 717 train ads, same schema {ad_id, T, R_true, raw}, same prompt v2 (causal, no R-leak).",
)
print("commit:", commit.commit_url if hasattr(commit, "commit_url") else commit)
print("dataset:", f"https://huggingface.co/datasets/{repo_id}")

# Sanity: re-fetch and list siblings
info = api.dataset_info(repo_id)
print("siblings after push:")
for s in info.siblings:
    print(" ", s.rfilename, "size=", s.size)
PY
