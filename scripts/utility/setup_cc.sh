#!/usr/bin/env bash
set -euxo pipefail
exec > /opt/dlami/nvme/setup_cc.log 2>&1
echo "[setup_cc] $(date) start"

# 1. Add Leon laptop pubkey to ubuntu authorized_keys
PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILq8DVvvohGXWwFQWx+aOyfNNYmPebxW98vpC/sHR0KL marvl@DN0a25a6b7.SUNet"
sudo mkdir -p /home/ubuntu/.ssh
echo "$PUBKEY" | sudo tee /home/ubuntu/.ssh/authorized_keys > /dev/null
sudo chown -R ubuntu:ubuntu /home/ubuntu/.ssh
sudo chmod 700 /home/ubuntu/.ssh
sudo chmod 600 /home/ubuntu/.ssh/authorized_keys

# 2. Enable + start sshd
sudo systemctl enable ssh
sudo systemctl start ssh
sudo systemctl status ssh --no-pager | head -5

# 3. Install Node 22 via NodeSource (Anthropic recommends Node 18+)
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
node --version
npm --version

# 4. Install Claude Code globally
sudo npm install -g @anthropic-ai/claude-code
which claude
claude --version || true

# 5. Make sure ubuntu user owns /opt/dlami/nvme/work so claude can edit there
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme/work 2>&1 | head -3 || true

echo "[setup_cc] $(date) DONE"
