#!/usr/bin/env bash
# Start Nava Vision AI Model Validation for developers (Docker + conda image).
# Exposes the app on all interfaces so teammates on the same Wi‑Fi can connect.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
export PORT HOST

usage() {
  cat <<EOF
Usage: ./dev-start.sh [up|down|logs|rebuild|local|url]

  up       Build (if needed) and start container in background (default)
  down     Stop and remove container
  logs     Follow container logs
  rebuild  Force rebuild image, then start
  local    Run without Docker using local conda env 'model-validation'
  url      Print localhost + Wi‑Fi URLs to share with developers
EOF
}

lan_ips() {
  # Prefer hostname -I (Linux); skip docker/tailscale-ish if many exist
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | tr ' ' '\n' | awk '
      $1 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ &&
      $1 !~ /^127\./ &&
      $1 !~ /^172\.(1[6-9]|2[0-9]|3[0-1])\./ &&
      $1 !~ /^100\./ { print; found=1 }
      END { if (!found) exit 1 }
    ' && return 0
  fi
  # Fallback: any non-loopback IPv4
  hostname -I 2>/dev/null | tr ' ' '\n' | awk '
    $1 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ && $1 !~ /^127\./ { print }
  '
}

print_urls() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Access URLs (same Wi‑Fi)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Local:    http://127.0.0.1:${PORT}"
  local ip
  local found=0
  # Prefer real NIC IPs (wlan/eth); skip docker/bridge/tailscale if possible
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
      # skip common docker bridges
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
  # Best-effort: open port on ufw if active (needs sudo; ignore failures)
  if command -v ufw >/dev/null 2>&1; then
    if sudo -n ufw status 2>/dev/null | grep -qi "Status: active"; then
      if ! sudo -n ufw status 2>/dev/null | grep -q "${PORT}/tcp"; then
        echo "Opening firewall port ${PORT}/tcp (ufw)…"
        sudo -n ufw allow "${PORT}/tcp" comment "Nava Model Validation" || true
      fi
    fi
  fi
}

need_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required. Install Docker, then retry." >&2
    exit 1
  fi
  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
  else
    echo "docker compose is required." >&2
    exit 1
  fi
}

cmd_up() {
  need_docker
  mkdir -p uploads outputs/recordings
  ensure_firewall
  echo "Starting Model Validation on ${HOST}:${PORT} (all network interfaces)…"
  echo "(GPU passthrough enabled via docker compose 'gpus: all')"
  "${COMPOSE[@]}" up -d --build
  print_urls
  echo "Check GPU: curl -s http://127.0.0.1:${PORT}/api/system"
  echo "Logs:      ./dev-start.sh logs"
  echo "Stop:      ./dev-start.sh down"
}

cmd_down() {
  need_docker
  "${COMPOSE[@]}" down
  echo "Stopped."
}

cmd_logs() {
  need_docker
  "${COMPOSE[@]}" logs -f model-validation
}

cmd_rebuild() {
  need_docker
  mkdir -p uploads outputs/recordings
  ensure_firewall
  "${COMPOSE[@]}" build --no-cache
  "${COMPOSE[@]}" up -d
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
