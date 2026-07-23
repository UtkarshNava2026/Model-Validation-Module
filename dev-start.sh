#!/usr/bin/env bash
# Start Nava Vision AI Model Validation for developers (Docker + conda image).
# Exposes the app on all interfaces so teammates on the same Wi‑Fi can connect.
#
# Works with:
#   - docker compose  (plugin v2)  OR
#   - docker-compose  (standalone) OR
#   - plain docker build/run       (fallback if compose is not installed)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
IMAGE="${IMAGE:-nava-vision-model-validation:latest}"
CONTAINER="${CONTAINER:-nava-model-validation}"
MAX_SESSIONS="${MAX_SESSIONS:-3}"
SESSION_TTL_SEC="${SESSION_TTL_SEC:-3600}"
export PORT HOST MAX_SESSIONS SESSION_TTL_SEC

# COMPOSE_MODE: plugin | standalone | docker
COMPOSE_MODE=""

usage() {
  cat <<EOF
Usage: ./dev-start.sh [up|down|logs|rebuild|local|url]

  up       Build (if needed) and start container in background (default)
  down     Stop and remove container
  logs     Follow container logs
  rebuild  Force rebuild image, then start
  local    Run without Docker using local conda env 'model-validation'
  url      Print localhost + Wi‑Fi URLs to share with developers

If 'docker compose' is missing, install it with:
  sudo apt-get update && sudo apt-get install -y docker-compose-plugin
Or the script will fall back to plain 'docker build/run'.
EOF
}

print_urls() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Access URLs (same Wi‑Fi)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Local:    http://127.0.0.1:${PORT}"
  local ip
  local found=0
  if command -v ip >/dev/null 2>&1; then
    while read -r ip; do
      [[ -z "$ip" ]] && continue
      echo "  Network:  http://${ip}:${PORT}   ← share this"
      found=1
    done < <(ip -4 -o addr show scope global 2>/dev/null | awk '
      $2 ~ /^(docker|br-|veth|virbr|tun|tap)/ { next }
      $2 == "tailscale0" { next }
      {
        split($4, a, "/")
        if (a[1] !~ /^127\./) print a[1]
      }
    ')
  fi
  if [[ "$found" -eq 0 ]]; then
    while read -r ip; do
      [[ -z "$ip" || "$ip" == 127.* ]] && continue
      [[ "$ip" == 172.17.* || "$ip" == 172.18.* ]] && continue
      echo "  Network:  http://${ip}:${PORT}   ← share this"
      found=1
    done < <(hostname -I 2>/dev/null | tr ' ' '\n')
  fi
  echo ""
  echo "  Share a Network URL with developers on the same Wi‑Fi."
  echo "  Listening on ${HOST}:${PORT} (all interfaces)."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
}

ensure_firewall() {
  if command -v ufw >/dev/null 2>&1; then
    if sudo -n ufw status 2>/dev/null | grep -qi "Status: active"; then
      if ! sudo -n ufw status 2>/dev/null | grep -q "${PORT}/tcp"; then
        echo "Opening firewall port ${PORT}/tcp (ufw)…"
        sudo -n ufw allow "${PORT}/tcp" comment "Nava Model Validation" || true
      fi
    fi
  fi
}

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_MODE="plugin"
    COMPOSE=(docker compose)
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_MODE="standalone"
    COMPOSE=(docker-compose)
    return 0
  fi
  COMPOSE_MODE="docker"
  COMPOSE=()
  return 0
}

need_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required. Install Docker, then retry." >&2
    exit 1
  fi
  detect_compose
  if [[ "$COMPOSE_MODE" == "docker" ]]; then
    echo "Note: 'docker compose' plugin not found — using plain docker build/run."
    echo "Optional (recommended): sudo apt-get install -y docker-compose-plugin"
    echo ""
  fi
}

gpu_args() {
  # Enable GPU if nvidia runtime / devices exist
  if docker info 2>/dev/null | grep -qi nvidia; then
    echo "--gpus all"
    return
  fi
  if [[ -e /dev/nvidia0 ]]; then
    echo "--gpus all"
    return
  fi
  echo ""
}

docker_run_up() {
  local rebuild="${1:-0}"
  mkdir -p uploads outputs/recordings sample_data

  if [[ "$rebuild" == "1" ]]; then
    docker build --no-cache -t "$IMAGE" .
  else
    docker build -t "$IMAGE" .
  fi

  # Replace existing container if present
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi

  # shellcheck disable=SC2046
  docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    -p "${HOST}:${PORT}:7860" \
    -e PORT=7860 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e "MAX_SESSIONS=${MAX_SESSIONS}" \
    -e "SESSION_TTL_SEC=${SESSION_TTL_SEC}" \
    -v "${ROOT}/uploads:/app/uploads" \
    -v "${ROOT}/outputs:/app/outputs" \
    -v "${ROOT}/sample_data:/app/sample_data:ro" \
    $(gpu_args) \
    "$IMAGE"
}

compose_up() {
  local rebuild="${1:-0}"
  if [[ "$rebuild" == "1" ]]; then
    "${COMPOSE[@]}" build --no-cache
    "${COMPOSE[@]}" up -d
  else
    "${COMPOSE[@]}" up -d --build
  fi
}

cmd_up() {
  need_docker
  mkdir -p uploads outputs/recordings
  ensure_firewall
  echo "Starting Model Validation on ${HOST}:${PORT} (all network interfaces)…"
  if [[ "$COMPOSE_MODE" == "docker" ]]; then
    docker_run_up 0
  else
    echo "(using ${COMPOSE_MODE} compose)"
    compose_up 0
  fi
  print_urls
  echo "Check GPU: curl -s http://127.0.0.1:${PORT}/api/system"
  echo "Logs:      ./dev-start.sh logs"
  echo "Stop:      ./dev-start.sh down"
}

cmd_down() {
  need_docker
  if [[ "$COMPOSE_MODE" == "docker" ]]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  else
    "${COMPOSE[@]}" down
  fi
  echo "Stopped."
}

cmd_logs() {
  need_docker
  if [[ "$COMPOSE_MODE" == "docker" ]]; then
    docker logs -f "$CONTAINER"
  else
    "${COMPOSE[@]}" logs -f model-validation
  fi
}

cmd_rebuild() {
  need_docker
  mkdir -p uploads outputs/recordings
  ensure_firewall
  if [[ "$COMPOSE_MODE" == "docker" ]]; then
    docker_run_up 1
  else
    compose_up 1
  fi
  print_urls
}

cmd_local() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Use ./dev-start.sh up for Docker, or install Miniconda." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "model-validation"; then
    echo "Creating conda env from environment.yml …"
    conda env create -f environment.yml
  fi
  conda activate model-validation
  if [[ ! -d third_party/YOLOX ]]; then
    bash scripts/setup_yolox.sh
  fi
  mkdir -p uploads outputs/recordings
  ensure_firewall
  print_urls
  echo "Starting locally (conda)…"
  python run.py
}

cmd_url() {
  print_urls
}

ACTION="${1:-up}"
case "$ACTION" in
  up) cmd_up ;;
  down) cmd_down ;;
  logs) cmd_logs ;;
  rebuild) cmd_rebuild ;;
  local) cmd_local ;;
  url) cmd_url ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: $ACTION" >&2
    usage
    exit 1
    ;;
esac
