#!/usr/bin/env bash
# Start Nava Vision AI Model Validation for developers (Docker + conda image).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-7860}"
export PORT

usage() {
  cat <<EOF
Usage: ./dev-start.sh [up|down|logs|rebuild|local]

  up       Build (if needed) and start container in background (default)
  down     Stop and remove container
  logs     Follow container logs
  rebuild  Force rebuild image, then start
  local    Run without Docker using local conda env 'model-validation'
EOF
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
  echo "Starting Model Validation on http://0.0.0.0:${PORT} …"
  "${COMPOSE[@]}" up -d --build
  echo ""
  echo "Ready:  http://localhost:${PORT}"
  echo "Logs:   ./dev-start.sh logs"
  echo "Stop:   ./dev-start.sh down"
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
  "${COMPOSE[@]}" build --no-cache
  "${COMPOSE[@]}" up -d
  echo "Rebuilt and running at http://localhost:${PORT}"
}

cmd_local() {
  # Prefer conda env for local (non-Docker) runs
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
  echo "Starting locally (conda) on http://0.0.0.0:${PORT} …"
  python run.py
}

ACTION="${1:-up}"
case "$ACTION" in
  up) cmd_up ;;
  down) cmd_down ;;
  logs) cmd_logs ;;
  rebuild) cmd_rebuild ;;
  local) cmd_local ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: $ACTION" >&2
    usage
    exit 1
    ;;
esac
