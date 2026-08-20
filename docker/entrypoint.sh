#!/usr/bin/env bash
# One entrypoint, four roles. Which role is decided by $1 so compose can share
# an image between `worker` and `beat` without a second Dockerfile.
set -euo pipefail

ROLE="${1:-api}"

log() { printf '%s entrypoint[%s] %s\n' "$(date -u +%H:%M:%S)" "$ROLE" "$*"; }

wait_for() {
  # $1 host, $2 port, $3 label. Bounded: 60 tries at 1s. A service that is not
  # up after a minute is a misconfiguration, and hanging forever hides it.
  local host="$1" port="$2" label="$3" i=0
  until python -c "import socket,sys; s=socket.socket(); s.settimeout(2); s.connect(('$host', $port))" 2>/dev/null; do
    i=$((i+1))
    if [ "$i" -ge 60 ]; then
      log "FATAL: $label at $host:$port did not become reachable in 60s"
      exit 1
    fi
    [ $((i % 10)) -eq 0 ] && log "still waiting for $label at $host:$port (${i}s)"
    sleep 1
  done
  log "$label is up at $host:$port"
}

# Parse host/port out of the URLs rather than requiring separate env vars, so
# there is exactly one source of truth per datastore. The scheme may carry a
# driver suffix (postgresql+psycopg), which urlsplit handles fine.
parse_hostport() {
  URL="$1" python - <<'PY'
import os
from urllib.parse import urlsplit
u = urlsplit(os.environ["URL"])
print(u.hostname or "", u.port or "")
PY
}

if [ -n "${DATABASE_URL:-}" ]; then
  read -r DB_HOST DB_PORT <<<"$(parse_hostport "$DATABASE_URL")"
  [ -n "$DB_HOST" ] && wait_for "$DB_HOST" "${DB_PORT:-5432}" postgres
fi
if [ -n "${REDIS_URL:-}" ]; then
  read -r RD_HOST RD_PORT <<<"$(parse_hostport "$REDIS_URL")"
  [ -n "$RD_HOST" ] && wait_for "$RD_HOST" "${RD_PORT:-6379}" redis
fi

case "$ROLE" in
  api)
    # Migrations run here and only here. Running them from the worker too would
    # race two `alembic upgrade head` processes against one another on a cold
    # start; Postgres advisory locks would survive it, but the logs would not
    # tell you which one won.
    log "running alembic upgrade head"
    alembic upgrade head
    log "starting uvicorn on :${PORT:-8000}"
    exec uvicorn app.main:app \
      --host 0.0.0.0 --port "${PORT:-8000}" \
      --workers "${API_WORKERS:-1}" \
      --proxy-headers --forwarded-allow-ips='*'
    ;;
  worker)
    # Queues are explicit: a 40-minute deepfake job must not be able to starve
    # 15-minute alert evaluation. Split the queues across replicas in prod.
    exec celery -A app.tasks.celery_app.celery worker \
      --loglevel="${CELERY_LOG_LEVEL:-info}" \
      --concurrency="${CELERY_CONCURRENCY:-2}" \
      --queues="${CELERY_QUEUES:-io,cpu,gpu,llm}" \
      --max-tasks-per-child="${CELERY_MAX_TASKS_PER_CHILD:-50}" \
      --hostname="worker@%h"
    ;;
  beat)
    exec celery -A app.tasks.celery_app.celery beat \
      --loglevel="${CELERY_LOG_LEVEL:-info}" \
      --schedule=/tmp/celerybeat-schedule
    ;;
  shell)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
