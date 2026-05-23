#!/bin/bash
set -e
source /home/ssm-user/work/venv/bin/activate
cp /tmp/new_readme.md /home/ssm-user/work/work-out/cot/README.md
python <<'PY'
from huggingface_hub import HfApi, login
from pathlib import Path
token_path = Path("/home/ssm-user/.cache/huggingface/token")
if token_path.exists():
    login(token=token_path.read_text().strip(), add_to_git_credential=False)

api = HfApi()
repo_id = "liangyuch/ttcc-cot"

# Upload both files in a single commit
ops = api.create_commit(
    repo_id=repo_id, repo_type="dataset",
    operations=[
        __import__("huggingface_hub").CommitOperationAdd(
            path_in_repo="cot_v2_causal_instruct.jsonl",
            path_or_fileobj="/home/ssm-user/work/work-out/cot/v3_pro_merged_final.jsonl",
        ),
        __import__("huggingface_hub").CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj="/home/ssm-user/work/work-out/cot/README.md",
        ),
    ],
    commit_message="v3 (Gemini-Pro teacher): 714 rows post-QC. README updated to reflect new teacher, prompt guardrails, and dropped IDs.",
)
print("commit:", getattr(ops, "commit_url", ops))

info = api.dataset_info(repo_id)
print("siblings after push:")
for s in info.siblings:
    print(" ", s.rfilename, "size=", s.size)
PY
