#!/usr/bin/env bash
set -euo pipefail

# Скрипт запускает MD для всех JSON-файлов в каталоге example/
# через API контейнера:
#  - для каждой системы делаются два запуска: с периодическими условиями и без них.
#
# Требования:
#  - запущен контейнер с API (см. compose.yaml), порт по умолчанию 65502
#  - утилиты curl и python3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$ROOT_DIR/example"
BASE_URL="${BASE_URL:-http://localhost:65502}"

if [[ ! -d "$EXAMPLE_DIR" ]]; then
  echo "Каталог с примерами не найден: $EXAMPLE_DIR" >&2
  exit 1
fi

JSON_FILES=("$EXAMPLE_DIR"/*.json)
if [[ ${#JSON_FILES[@]} -eq 0 ]]; then
  echo "В каталоге $EXAMPLE_DIR нет JSON-файлов." >&2
  exit 1
fi

echo "Будут запущены MD-симуляции для файлов:"
for f in "${JSON_FILES[@]}"; do
  echo "  - $(basename "$f")"
done
echo

# Параметры MD:
# 100000 шагов при шаге 0.5 fs => 50 ps, dump каждые 50 fs
TIME_PS=50.0
STEP_FS=0.5
DUMP_FS=5.0

for json_path in "${JSON_FILES[@]}"; do
  base_name="$(basename "$json_path")"
  echo "============================================================"
  echo "Система: $base_name"

  # Кодируем JSON в base64 для inline-передачи
  B64_CONTENT="$(python3 - <<PY
from pathlib import Path
import base64
p = Path("$json_path")
data = p.read_bytes()
print(base64.b64encode(data).decode("ascii"))
PY
)"

  for periodic_flag in false true; do
    suffix="$([[ "$periodic_flag" == "true" ]] && echo "per" || echo "nper")"
    job_id="${base_name%.json}_md_${suffix}"

    echo
    echo "==> Запуск MD (${suffix}) для $base_name, job_id=$job_id"

    REQ_FILE="$(mktemp)"
    cat > "$REQ_FILE" <<EOF
{
  "mode": "MD",
  "job_id": "$job_id",
  "input_files_inline": [
    {
      "name": "$base_name",
      "content_base64": "$B64_CONTENT"
    }
  ],
  "md": {
    "periodic": $periodic_flag,
    "time_ps": $TIME_PS,
    "step_fs": $STEP_FS,
    "dump_fs": $DUMP_FS,
    "temperature_k": 298.15,
    "nvt": true
  }
}
EOF

    echo "   -> POST /run ..."
    RESP="$(curl -sS -X POST "$BASE_URL/run" \
      -H "Content-Type: application/json" \
      --data-binary "@$REQ_FILE")" || {
      echo "Ошибка запроса /run для job_id=$job_id" >&2
      rm -f "$REQ_FILE"
      continue
    }

    echo "Ответ /run:"
    echo "$RESP"

    rm -f "$REQ_FILE"
  done
done

echo
echo "Все запросы /run для примерных систем отправлены."

