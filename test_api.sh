#!/usr/bin/env bash
set -euo pipefail

# Тестовый скрипт для проверки API:
#  - POST /run (MD, непериодический расчёт)
#  - опрос /status/{job_id} до завершения
#  - POST /analyze по полученному архиву
#
# Требования:
#  - запущен контейнер с API (см. compose.yaml), порт по умолчанию 65502
#  - утилиты curl и python3
#  - jq (опционально, но рекомендуется; без него будет использован python для разбора JSON)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL:-http://localhost:65502}"

JSON_EXAMPLE="$ROOT_DIR/examples/json/system_example.json"
if [[ ! -f "$JSON_EXAMPLE" ]]; then
  echo "Не найден файл примера системы: $JSON_EXAMPLE" >&2
  exit 1
fi

echo "==> Кодирование примера системы в base64 (inline-файл)..."
B64_CONTENT="$(python3 - <<PY
from pathlib import Path
import base64
p = Path("$JSON_EXAMPLE")
data = p.read_bytes()
print(base64.b64encode(data).decode("ascii"))
PY
)"

REQ_FILE="$(mktemp)"
cat > "$REQ_FILE" <<EOF
{
  "mode": "MD",
  "job_id": "api_test_md",
  "input_files_inline": [
    {
      "name": "system_example.json",
      "content_base64": "$B64_CONTENT"
    }
  ],
  "md": {
    "periodic": false,
    "time_ps": 1.0,
    "step_fs": 0.5,
    "dump_fs": 1.0,
    "temperature_k": 298.15,
    "nvt": true
  }
}
EOF

echo "==> Отправка запроса на запуск MD через /run..."
RAW_RUN_RESP="$(curl -sS -X POST "$BASE_URL/run" \
  -H "Content-Type: application/json" \
  --data-binary "@$REQ_FILE")"

echo "Ответ /run:"
echo "$RAW_RUN_RESP"

parse_json() {
  local json="$1"
  local key="$2"
  if command -v jq >/dev/null 2>&1; then
    printf '%s\n' "$json" | jq -r ".$key"
  else
    python3 - <<PY
import json, sys
data = json.loads(sys.stdin.read())
print(data.get("$key", ""))
PY
  fi
}

JOB_ID="$(parse_json "$RAW_RUN_RESP" "job_id")"
ARCHIVE_PATH_ABS="$(parse_json "$RAW_RUN_RESP" "archive_path")"
STATUS="$(parse_json "$RAW_RUN_RESP" "status")"

if [[ -z "$JOB_ID" || "$JOB_ID" == "null" ]]; then
  echo "Не удалось извлечь job_id из ответа /run" >&2
  exit 1
fi

echo "==> job_id: $JOB_ID, status: $STATUS"

echo "==> Опрос статуса через /status/${JOB_ID}..."
while true; do
  RAW_STATUS="$(curl -sS "$BASE_URL/status/$JOB_ID")" || {
    echo "Ошибка запроса /status/$JOB_ID" >&2
    exit 1
  }
  STATUS_VAL="$(parse_json "$RAW_STATUS" "status")"
  MSG_VAL="$(parse_json "$RAW_STATUS" "message")"
  ARCHIVE_STATUS_PATH="$(parse_json "$RAW_STATUS" "archive_path")"

  echo "Статус: $STATUS_VAL, message: ${MSG_VAL:-""}"

  if [[ "$STATUS_VAL" != "running" && "$STATUS_VAL" != "queued" ]]; then
    ARCHIVE_PATH_ABS="${ARCHIVE_STATUS_PATH:-$ARCHIVE_PATH_ABS}"
    break
  fi

  sleep 2
done

if [[ -z "$ARCHIVE_PATH_ABS" || "$ARCHIVE_PATH_ABS" == "null" ]]; then
  echo "Архив не найден в ответах API, пропускаем /analyze." >&2
  exit 0
fi

echo "==> Итоговый путь к архиву (внутри контейнера): $ARCHIVE_PATH_ABS"

# Для /analyze нужен путь относительно /data
ARCHIVE_REL="$ARCHIVE_PATH_ABS"
ARCHIVE_REL="\${ARCHIVE_REL#/data/}"

REQ_ANALYZE="$(mktemp)"
cat > "$REQ_ANALYZE" <<EOF
{
  "archive_path": "$ARCHIVE_REL"
}
EOF

echo "==> Запуск анализа архива через /analyze..."
RAW_ANALYZE_RESP="$(curl -sS -X POST "$BASE_URL/analyze" \
  -H "Content-Type: application/json" \
  --data-binary "@$REQ_ANALYZE")"

echo "Ответ /analyze:"
echo "$RAW_ANALYZE_RESP"

REPORT_DIR="$(parse_json "$RAW_ANALYZE_RESP" "report_dir")"
echo "==> report_dir (внутри контейнера): ${REPORT_DIR:-<не указан>}"

echo "Готово."

