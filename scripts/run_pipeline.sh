#!/usr/bin/env bash
set -euo pipefail

# Аргументы:
#  1) JSON с описанием системы (как раньше для build_system_from_json.py)
#  2) Время MD (ps)
#  3) Шаг интегрирования (fs)
#  4) Шаг дампа (fs)
#
# Пример:
#  scripts/run_pipeline.sh examples/json/system_example.json 100 0.5 10
#

if [ "$#" -lt 4 ]; then
  echo "Использование: $0 system.json time_ps step_fs dump_fs" >&2
  exit 1
fi

JSON_FILE="$1"
TIME_PS="$2"
STEP_FS="$3"
DUMP_FS="$4"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVES_DIR="$ROOT_DIR/archives"
mkdir -p "$ARCHIVES_DIR"

# Имя системы по JSON (без пути и расширения)
SYSTEM_NAME="$(basename "$JSON_FILE" .json)"

# Вытащим краткое описание состава из JSON (name+count),
# чтобы сформировать суффикс вида 100X_50Y и т.п.
COMPONENT_TAGS="$(
  python3 - "$JSON_FILE" <<'PY'
import json, sys, pathlib

path = pathlib.Path(sys.argv[1])
data = json.load(path.open())
parts = []
for item in data:
    name = str(item.get("name", "X"))
    count = int(item.get("count", 0))
    parts.append(f"{count}{name}")
print("_".join(parts))
PY
)"

RUN_TAG="${TIME_PS}ps_${STEP_FS}fs_${DUMP_FS}fs_${COMPONENT_TAGS}"

RUN_DIR="$ROOT_DIR/runs/${SYSTEM_NAME}/${RUN_TAG}"
OPT_DIR="$RUN_DIR/opt"
MD_DIR="$RUN_DIR/md"

mkdir -p "$OPT_DIR" "$MD_DIR"

export OMP_STACKSIZE="32G"

cd "$ROOT_DIR"

echo "==> Генерация и упаковка системы..."
python3 lib/build_system_from_json.py "$JSON_FILE" -o "$OPT_DIR/system.xyz"

echo "==> Геометрическая оптимизация (xtb)..."
cd "$OPT_DIR"
"$ROOT_DIR/xtb-dist/bin/xtb" system.xyz --opt > log
cd "$ROOT_DIR"

echo "==> Конвертация в Turbomole coord..."
python3 lib/xyz_to_turbomole_coord.py "$OPT_DIR/xtbopt.xyz" -o "$MD_DIR/coord"

echo "==> Добавление MD-блока и запуск MD..."
cat >> "$MD_DIR/coord" <<EOF
\$md
   temp=298.15 # in K
   time= $TIME_PS # in ps
   dump= $DUMP_FS  # in fs
   step=  $STEP_FS  # in fs
   velo=false
   nvt =true
   hmass=1
   shake=0
   sccacc=2.0
\$end
EOF
cd "$MD_DIR"
"$ROOT_DIR/xtb-dist/bin/xtb" coord --gfnff --md

echo "==> Конвертация траектории в XYZ..."
sed -E 's/^([hocns])/\U\1/' xtb.trj > result.xyz
cd "$ROOT_DIR"

echo "==> Конвертация XYZ -> CIF..."
python3 lib/xyz2cif.py --xyz "$MD_DIR/result.xyz" --coord "$MD_DIR/coord" -o "$RUN_DIR/traj.cif"

echo "==> Постобработка MD (CSV для отчёта)..."
python3 lib/postprocess_md.py \
  --trj "$MD_DIR/xtb.trj" \
  --coord "$MD_DIR/coord" \
  --dump-fs "$DUMP_FS" \
  --out-dir "$RUN_DIR"

echo "==> Архивация результата..."
ARCHIVE_NAME="${RUN_TAG}.tar.gz"
cd "$ROOT_DIR/runs/${SYSTEM_NAME}"
tar -czf "$ARCHIVES_DIR/$ARCHIVE_NAME" "$RUN_TAG"

echo "Готово. Архив: $ARCHIVES_DIR/$ARCHIVE_NAME"

