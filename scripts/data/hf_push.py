"""Push the v2 causal CoT JSONL + README to liangyuch/ttcc-cot on HF Hub."""
from huggingface_hub import HfApi, create_repo

REPO_ID = "liangyuch/ttcc-cot"
JSONL = "/home/ssm-user/work/work-out/cot/v2_full_instruct_merged.jsonl"
README = "/tmp/HF_README.md"
TOKEN = open("/tmp/hf_token").read().strip()

api = HfApi(token=TOKEN)

print("creating repo (idempotent) ...")
create_repo(REPO_ID, repo_type="dataset", token=TOKEN, exist_ok=True, private=False)

print("uploading JSONL ...")
api.upload_file(
    path_or_fileobj=JSONL,
    path_in_repo="cot_v2_causal_instruct.jsonl",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add v2 causal CoT (717 rows, Qwen3-Omni-30B-A3B-Instruct teacher, video-only prompt)",
)

print("uploading README ...")
api.upload_file(
    path_or_fileobj=README,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add README with schema + generation provenance",
)

print(f"OK: https://huggingface.co/datasets/{REPO_ID}")
