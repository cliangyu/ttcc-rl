#!/bin/bash
source /home/ssm-user/work/venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/home/ssm-user/work/vizzy-sa.json
python <<'PY'
from google.cloud import storage
client = storage.Client(project="vizzylabs-ai-prod")
bucket = client.bucket("ttcc-cot-staging-191aab")
blobs = list(bucket.list_blobs(prefix="v3-pro/oversize/"))
print(f"deleting {len(blobs)} blobs from gs://ttcc-cot-staging-191aab/v3-pro/oversize/")
for b in blobs:
    b.delete()
# Delete the bucket itself (now empty)
remaining = list(bucket.list_blobs())
print(f"blobs remaining: {len(remaining)}")
if not remaining:
    bucket.delete()
    print("bucket deleted")
PY
