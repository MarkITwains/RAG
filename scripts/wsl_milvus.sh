#!/usr/bin/env bash
# Manage the existing Milvus database under WSL. Run as root for Docker access.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$EUID" -ne 0 ]; then
  printf 'Run this script as WSL root (wsl -d pure2404 -u root).\n' >&2
  exit 1
fi

RUNTIME_USER="${PCB_RAG_USER:-$(stat -c %U "$ROOT_DIR")}"
RUNTIME_HOME="$(getent passwd "$RUNTIME_USER" | cut -d: -f6)"
if [ -z "$RUNTIME_HOME" ] || [ "$RUNTIME_USER" = root ]; then
  printf 'Set PCB_RAG_USER to the Linux account that owns this project.\n' >&2
  exit 1
fi

export DOCKER_VOLUME_DIRECTORY="${PCB_RAG_MILVUS_DIR:-$RUNTIME_HOME/.local/share/pcb-rag/milvus}"
COMPOSE=(docker compose --progress quiet --project-name pcb-rag-wsl
  -f "$ROOT_DIR/docker/milvus/docker-compose.yml"
  -f "$ROOT_DIR/docker/milvus/docker-compose.wsl.yml")

case "${1:-start}" in
  start)
    if [ ! -d "$DOCKER_VOLUME_DIRECTORY/volumes/etcd" ] || \
       [ ! -d "$DOCKER_VOLUME_DIRECTORY/volumes/minio" ]; then
      printf 'Expected the existing database snapshot under %s/volumes.\n' "$DOCKER_VOLUME_DIRECTORY" >&2
      exit 1
    fi
    systemctl start docker
    "${COMPOSE[@]}" up -d --wait --wait-timeout 180
    ;;
  stop)
    "${COMPOSE[@]}" stop
    ;;
  status)
    "${COMPOSE[@]}" ps
    ;;
  logs)
    "${COMPOSE[@]}" logs --tail "${LOG_LINES:-60}" standalone
    ;;
  *)
    printf 'Usage: bash scripts/wsl_milvus.sh {start|stop|status|logs}\n' >&2
    exit 2
    ;;
esac
