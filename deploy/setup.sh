#!/usr/bin/env bash
# One-shot setup of Certus on a fresh Ubuntu 24.04 VM, x64 or Arm64. Sized for 1 GB of RAM
# (Azure B1s or B2pts_v2).
#
#   git clone https://github.com/NikunjKaushik20/Certus.git ~/certus
#   bash ~/certus/deploy/setup.sh certus-demo.centralindia.cloudapp.azure.com
#
# Serves the web app and the API from one HTTPS origin with Caddy (automatic Let's Encrypt), runs
# the API on onnxruntime with no torch installed, and restarts it on boot or crash. Safe to re-run:
# after a `git pull`, running it again rebuilds and restarts.
#
# API keys: the sign-in page offers the demo keys, so they stay unless CERTUS_API_KEYS is set,
# e.g. CERTUS_API_KEYS="k1:technician,k2:ophthalmologist,k3:admin" bash setup.sh <host>.
# Anyone with the URL can sign in with a demo key: do not upload real patient photos.
set -euo pipefail

HOST=${1:?usage: setup.sh <hostname pointing at this VM>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
USER_NAME=$(id -un)
WEB_ROOT=/var/www/certus

echo "== swap (a safety net; a graded eye peaks near 430 MB)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv caddy curl ca-certificates >/dev/null
if ! node -v 2>/dev/null | grep -qE '^v(2[2-9]|[3-9][0-9])'; then   # Vite 8 needs Node 22
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null
  sudo apt-get install -y -qq nodejs >/dev/null
fi

echo "== API (onnxruntime, no torch)"
python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r "$REPO/api/requirements-server.txt"

echo "== web app"
cd "$REPO/web"
npm ci --no-audit --no-fund --loglevel=error
VITE_API_BASE="" npm run build              # "" = same origin: Caddy forwards /v1 to the API
sudo mkdir -p "$WEB_ROOT"
sudo rm -rf "${WEB_ROOT:?}"/*
sudo cp -r dist/. "$WEB_ROOT/"

echo "== service"
KEYS_LINE=""
if [ -n "${CERTUS_API_KEYS:-}" ]; then
  KEYS_LINE="Environment=CERTUS_API_KEYS=${CERTUS_API_KEYS}"
fi
sudo tee /etc/systemd/system/certus.service >/dev/null <<EOF
[Unit]
Description=Certus API
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${REPO}/api
Environment=CERTUS_RUNTIME=onnx
Environment=CERTUS_DEVICE=cpu
Environment=CERTUS_THREADS=$(nproc)
Environment=MALLOC_ARENA_MAX=2
${KEYS_LINE}
ExecStart=${REPO}/.venv/bin/uvicorn certus_api.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable certus >/dev/null 2>&1
sudo systemctl restart certus

echo "== HTTPS front door"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
${HOST} {
    encode gzip
    @api path /v1/* /healthz /docs /openapi.json
    handle @api {
        reverse_proxy 127.0.0.1:8000
    }
    handle {
        root * ${WEB_ROOT}
        try_files {path} /index.html
        file_server
    }
}
EOF
sudo systemctl reload caddy || sudo systemctl restart caddy

echo "== waiting for the API"
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS http://127.0.0.1:8000/healthz && echo
echo "Done: https://${HOST}  (the certificate can take a minute on first run)"
echo "Logs: journalctl -u certus -f"
