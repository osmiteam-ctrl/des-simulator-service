#!/usr/bin/env bash
set -euo pipefail

# Тест периодического и непериодического MD через md_runner.run_md
#
# Требования:
#  - собранный xtb в ./xtb-dist/bin/xtb
#  - примеры систем в examples/json/system_example.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JSON_SYSTEM="$ROOT_DIR/examples/json/system_example.json"
TIME_PS=1.0
STEP_FS=0.5
DUMP_FS=1.0

echo "==> Непериодический MD: xtb system.xyz --md"
python3 - <<PY
from pathlib import Path
from lib import md_runner

root = Path("$ROOT_DIR")
json_system = Path("$JSON_SYSTEM")
output_root = root / "test_output"

archive_np = md_runner.run_md(
    json_system=json_system,
    time_ps=$TIME_PS,
    step_fs=$STEP_FS,
    dump_fs=$DUMP_FS,
    periodic=False,
    output_root=output_root,
    job_id="test_nonperiodic",
    temperature_k=298.15,
    nvt=True,
)
print("Non-periodic archive:", archive_np)
PY

echo
echo "==> Периодический MD: xtb coord --gfnff --md"
python3 - <<PY
from pathlib import Path
from lib import md_runner

root = Path("$ROOT_DIR")
json_system = Path("$JSON_SYSTEM")
output_root = root / "test_output"

archive_p = md_runner.run_md(
    json_system=json_system,
    time_ps=$TIME_PS,
    step_fs=$STEP_FS,
    dump_fs=$DUMP_FS,
    periodic=True,
    output_root=output_root,
    job_id="test_periodic",
    temperature_k=298.15,
    nvt=True,
)
print("Periodic archive:", archive_p)
PY

echo
echo "Готово. Архивы тестовых запусков находятся в: $ROOT_DIR/test_output/archives"