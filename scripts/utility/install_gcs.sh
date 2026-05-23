#!/bin/bash
source /home/ssm-user/work/venv/bin/activate
pip install --quiet google-cloud-storage
python -c "from google.cloud import storage; print('storage ok')"
