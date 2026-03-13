from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
import uuid
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
XTB_BIN = ROOT_DIR / "xtb-dist" / "bin" / "xtb"


class SimulationError(RuntimeError):
    pass


def _ensure_xtb_exists() -> None:
    if not XTB_BIN.is_file():
        raise SimulationError(f"XTB binary not found at {XTB_BIN}")


def _prepare_workdir(
    job_id: Optional[str],
    input_paths: Optional[Iterable[Path]],
    inline_files: Optional[Iterable[Tuple[str, str]]],
    base_input_dir: Path,
) -> Tuple[str, Path]:
    """
    Создаёт рабочую директорию задачи и заполняет её входными файлами.

    input_paths: относительные пути внутри base_input_dir
    inline_files: (name, content_base64)
    """
    job_id_real = job_id or str(uuid.uuid4())
    work_root = ROOT_DIR / "api_runs"
    work_dir = work_root / job_id_real
    work_dir.mkdir(parents=True, exist_ok=True)

    # Копируем файлы по путям
    if input_paths:
        for rel in input_paths:
            src = (base_input_dir / rel).resolve()
            if not src.is_file():
                raise SimulationError(f"Input file not found: {src}")
            dst = work_dir / src.name
            shutil.copy2(src, dst)

    # Сохраняем inline‑файлы
    if inline_files:
        for name, b64 in inline_files:
            dst = work_dir / name
            data = base64.b64decode(b64)
            dst.write_bytes(data)

    return job_id_real, work_dir


def run_opt(
    json_system: Path,
    periodic: bool,
    output_root: Path,
    job_id: Optional[str] = None,
) -> Path:
    """
    Запуск геометрической оптимизации XTB по JSON‑описанию системы.

    На выходе возвращает путь к архиву результатов.
    """
    _ensure_xtb_exists()

    job_id_real = job_id or str(uuid.uuid4())
    system_name = json_system.stem

    run_dir = output_root / "runs" / system_name / job_id_real
    opt_dir = run_dir / "opt"
    opt_dir.mkdir(parents=True, exist_ok=True)

    env = dict(**{"OMP_STACKSIZE": "32G"})

    # 1) Генерация системы
    build_script = ROOT_DIR / "lib" / "build_system_from_json.py"
    cmd_build = [sys.executable, str(build_script), str(json_system), "-o", str(opt_dir / "system.xyz")]
    subprocess.run(cmd_build, cwd=ROOT_DIR, env=env, check=True)

    # 2) OPT в XTB
    cmd_xtb = [str(XTB_BIN), "system.xyz", "--opt"]
    # Периодические условия в XTB для OPT (если нужно) можно будет донастроить здесь
    subprocess.run(cmd_xtb, cwd=opt_dir, env=env, check=True)

    # Архивация результатов OPT
    archives_dir = output_root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / f"{job_id_real}.opt.tar.gz"

    subprocess.run(
        ["tar", "-czf", str(archive_path), job_id_real],
        cwd=run_dir.parent,
        check=True,
    )

    return archive_path


def run_md(
    json_system: Path,
    time_ps: float,
    step_fs: float,
    dump_fs: float,
    periodic: bool,
    output_root: Path,
    job_id: Optional[str] = None,
    temperature_k: float = 298.15,
    nvt: bool = True,
) -> Path:
    """
    Запуск MD по JSON‑описанию системы.
    Логика в целом повторяет scripts/run_pipeline.sh, но реализована на Python.
    """
    _ensure_xtb_exists()

    job_id_real = job_id or f"{time_ps}ps_{step_fs}fs_{dump_fs}fs"
    system_name = json_system.stem

    run_dir = output_root / "runs" / system_name / job_id_real
    opt_dir = run_dir / "opt"
    md_dir = run_dir / "md"
    opt_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    env = dict(**{"OMP_STACKSIZE": "32G"})

    # 1) Генерация системы
    build_script = ROOT_DIR / "lib" / "build_system_from_json.py"
    subprocess.run(
        [sys.executable, str(build_script), str(json_system), "-o", str(opt_dir / "system.xyz")],
        cwd=ROOT_DIR,
        env=env,
        check=True,
    )

    # 2) Геометрическая оптимизация
    subprocess.run(
        [str(XTB_BIN), "system.xyz", "--opt"],
        cwd=opt_dir,
        env=env,
        check=True,
    )

    # 3) Конвертация в coord (используется для периодического MD и анализа)
    coord_script = ROOT_DIR / "lib" / "xyz_to_turbomole_coord.py"
    subprocess.run(
        [sys.executable, str(coord_script), str(opt_dir / "xtbopt.xyz"), "-o", str(md_dir / "coord")],
        cwd=ROOT_DIR,
        env=env,
        check=True,
    )

    # 4) Добавление MD‑блока в coord (для периодического MD)
    md_block_lines = [
        "$md",
        f"   temp= {temperature_k} # in K",
        f"   time= {time_ps} # in ps",
        f"   dump= {dump_fs}  # in fs",
        f"   step=  {step_fs}  # in fs",
        "   velo=false",
        f"   nvt ={'true' if nvt else 'false'}",
        "   hmass=1",
        "   shake=0",
        "   sccacc=2.0",
    ]
    if periodic:
        md_block_lines.append("   periodic=true")
    md_block_lines.append("$end")

    with (md_dir / "coord").open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(md_block_lines) + "\n")

    # 5) Запуск MD
    if periodic:
        # Периодический расчёт: coord + GFN-FF
        cmd_md = [str(XTB_BIN), "coord", "--gfnff", "--md"]
        md_cwd = md_dir
    else:
        # Непериодический расчёт: обычный MD по системе в формате XYZ
        # Копируем оптимизированную геометрию в рабочую директорию MD
        src_xyz = opt_dir / "xtbopt.xyz"
        dst_xyz = md_dir / "system.xyz"
        shutil.copy2(src_xyz, dst_xyz)
        cmd_md = [str(XTB_BIN), "system.xyz", "--md"]
        md_cwd = md_dir

    subprocess.run(
        cmd_md,
        cwd=md_cwd,
        env=env,
        check=True,
    )

    # 6) Конвертация траектории в XYZ
    trj_path = md_dir / "xtb.trj"
    result_xyz = md_dir / "result.xyz"
    with trj_path.open("r", encoding="utf-8", errors="ignore") as fin, result_xyz.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            # Заменяем первые буквы атомов на заглавные (как в bash‑скрипте)
            fout.write(line.replace("h", "H").replace("o", "O").replace("c", "C").replace("n", "N").replace("s", "S"))

    # 7) XYZ -> CIF
    xyz2cif_script = ROOT_DIR / "lib" / "xyz2cif.py"
    subprocess.run(
        [
            sys.executable,
            str(xyz2cif_script),
            "--xyz",
            str(result_xyz),
            "--coord",
            str(md_dir / "coord"),
            "-o",
            str(run_dir / "traj.cif"),
        ],
        cwd=ROOT_DIR,
        env=env,
        check=True,
    )

    # 8) Постобработка MD
    postprocess_script = ROOT_DIR / "lib" / "postprocess_md.py"
    subprocess.run(
        [
            sys.executable,
            str(postprocess_script),
            "--trj",
            str(md_dir / "xtb.trj"),
            "--coord",
            str(md_dir / "coord"),
            "--dump-fs",
            str(dump_fs),
            "--out-dir",
            str(run_dir),
        ],
        cwd=ROOT_DIR,
        env=env,
        check=True,
    )

    # 9) Архивация результата
    archives_dir = output_root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / f"{job_id_real}.md.tar.gz"

    subprocess.run(
        ["tar", "-czf", str(archive_path), job_id_real],
        cwd=run_dir.parent,
        check=True,
    )

    return archive_path


