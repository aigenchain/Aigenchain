#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/aigenchain-ui.service"

if [ ! -f "$SERVICE_FILE" ]; then
  echo "Error: aigenchain-ui.service not found in $SCRIPT_DIR"
  exit 1
fi

echo "Installing Aigenchain UI service..."
echo "Make sure you've edited aigenchain-ui.service with your username and paths first!"
echo ""

sudo cp "$SERVICE_FILE" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aigenchain-ui
sudo systemctl start aigenchain-ui
sudo systemctl status aigenchain-ui
