from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal, Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lib import md_runner
from analyze_archive import main as analyze_archive_main


ROOT_DIR = Path(__file__).resolve().parent


class InputFileInline(BaseModel):
    name: str = Field(..., description="Имя файла, которое будет использовано при сохранении во временную директорию задачи")
    content_base64: str = Field(..., description="Содержимое файла в base64")


class OptParams(BaseModel):
    periodic: bool = Field(False, description="Периодические граничные условия для OPT")
    # Здесь можно добавить дополнительные параметры оптимизации при расширении функционала


class MdParams(BaseModel):
    periodic: bool = Field(False, description="Периодические граничные условия для MD")
    time_ps: float = Field(..., gt=0, description="Время MD в пикосекундах")
    step_fs: float = Field(..., gt=0, description="Шаг интегрирования в фемтосекундах")
    dump_fs: float = Field(..., gt=0, description="Период дампа траектории в фемтосекундах")
    temperature_k: float = Field(298.15, gt=0, description="Температура в Кельвинах")
    nvt: bool = Field(True, description="Использовать NVT ансамбль")
    # При необходимости сюда можно добавить и другие параметры блока $md


class SimulationRequest(BaseModel):
    mode: Literal["OPT", "MD"] = Field(..., description="Тип расчёта: OPT или MD")
    job_id: Optional[str] = Field(None, description="Необязательный пользовательский идентификатор задачи")

    # Вариант 1: указать путь к JSON/другим входным файлам во входном каталоге
    input_paths: Optional[List[str]] = Field(
        None,
        description="Список относительных путей к входным файлам во входном смонтированном каталоге",
    )

    # Вариант 2: передать файлы целиком
    input_files_inline: Optional[List[InputFileInline]] = Field(
        None,
        description="Список файлов, переданных в теле запроса и сохраняемых во временную директорию задачи",
    )

    opt: Optional[OptParams] = Field(None, description="Параметры оптимизации, используются при mode=OPT")
    md: Optional[MdParams] = Field(None, description="Параметры MD, используются при mode=MD")


class SimulationResponse(BaseModel):
    status: Literal["done", "error"]
    job_id: str
    archive_path: Optional[str] = Field(
        None, description="Путь к архиву с результатами внутри примонтированного каталога"
    )
    message: Optional[str] = None


class AnalyzeRequest(BaseModel):
    archive_path: str = Field(..., description="Путь к архиву с результатами внутри примонтированного каталога /data")


class AnalyzeResponse(BaseModel):
    status: Literal["done", "error"]
    report_dir: Optional[str] = None
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    status: Literal["queued", "running", "done", "error", "unknown"]
    job_id: str
    mode: Optional[Literal["OPT", "MD"]] = None
    archive_path: Optional[str] = None
    message: Optional[str] = None


app = FastAPI(title="MD/OPT XTB simulation API", docs_url="/docs", redoc_url="/redoc")


@app.post("/run", response_model=SimulationResponse)
def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """
    Запуск OPT или MD по JSON‑описанию.

    Пока что это только каркас: фактическая логика запуска XTB и упаковки
    будет реализована в lib.md_runner и подключена сюда.
    """
    # Простая проверка соответствия mode и параметров
    if request.mode == "OPT" and request.opt is None:
        raise HTTPException(status_code=400, detail="Для mode=OPT необходимо заполнить поле 'opt'.")
    if request.mode == "MD" and request.md is None:
        raise HTTPException(status_code=400, detail="Для mode=MD необходимо заполнить поле 'md'.")

    output_root = Path("/data").resolve()
    status_dir = output_root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    input_base = output_root / "input"
    input_paths = [Path(p) for p in (request.input_paths or [])]
    inline_files = []
    for f in request.input_files_inline or []:
        inline_files.append((f.name, f.content_base64))

    if not input_paths and not inline_files:
        raise HTTPException(status_code=400, detail="Нужно указать хотя бы один входной файл (input_paths или input_files_inline).")

    # Генерируем/фиксируем job_id
    job_id = request.job_id or str(uuid.uuid4())

    # Пока предполагаем один основной JSON‑файл системы
    json_path: Optional[Path] = None
    if input_paths:
        # Берём первый путь как основной JSON
        json_path = (input_base / input_paths[0]).resolve()
    elif inline_files:
        # Сохраним во временную директорию и используем первый файл как JSON
        job_id, tmp_dir = md_runner._prepare_workdir(
            job_id,
            input_paths=None,
            inline_files=inline_files,
            base_input_dir=input_base,
        )
        json_path = tmp_dir / inline_files[0][0]
    else:
        raise HTTPException(status_code=400, detail="Не удалось определить входной JSON.")

    if not json_path.is_file():
        raise HTTPException(status_code=400, detail=f"Файл системы не найден: {json_path}")

    status_file = status_dir / f"{job_id}.json"

    def save_status(data: dict) -> None:
        status_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # Инициализируем статус задачи
    status_payload: dict = {
        "status": "running",
        "job_id": job_id,
        "mode": request.mode,
        "archive_path": None,
        "message": "Simulation is running.",
    }
    save_status(status_payload)

    try:
        if request.mode == "OPT":
            params = request.opt  # type: ignore[assignment]
            archive = md_runner.run_opt(
                json_system=json_path,
                periodic=params.periodic,  # type: ignore[union-attr]
                output_root=output_root,
                job_id=job_id,
            )
        else:
            params = request.md  # type: ignore[assignment]
            archive = md_runner.run_md(
                json_system=json_path,
                time_ps=params.time_ps,  # type: ignore[union-attr]
                step_fs=params.step_fs,  # type: ignore[union-attr]
                dump_fs=params.dump_fs,  # type: ignore[union-attr]
                periodic=params.periodic,  # type: ignore[union-attr]
                output_root=output_root,
                job_id=job_id,
                temperature_k=params.temperature_k,  # type: ignore[union-attr]
                nvt=params.nvt,  # type: ignore[union-attr]
            )
    except md_runner.SimulationError as e:
        status_payload.update({"status": "error", "message": str(e), "archive_path": None})
        save_status(status_payload)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        status_payload.update({"status": "error", "message": f"Unexpected error: {e}", "archive_path": None})
        save_status(status_payload)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    status_payload.update(
        {
            "status": "done",
            "archive_path": str(archive),
            "message": "Simulation finished successfully.",
        }
    )
    save_status(status_payload)

    return SimulationResponse(
        status="done",
        job_id=job_id,
        archive_path=str(archive),
        message="Simulation finished successfully.",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_archive(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Запуск analyze_archive.py внутри контейнера по уже готовому архиву.
    """
    archive_full = Path("/data").resolve() / request.archive_path
    if not archive_full.is_file():
        raise HTTPException(status_code=400, detail=f"Архив не найден: {archive_full}")

    # Каталог отчёта будет создан самим analyze_archive.py по имени архива
    try:
        analyze_archive_main([str(archive_full)])
    except SystemExit as e:
        if e.code not in (0, None):
            raise HTTPException(status_code=500, detail=f"analyze_archive завершился с кодом {e.code}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе архива: {e}")

    # По соглашению analyze_archive кладёт отчёт рядом с архивом с суффиксом _report
    base = archive_full
    if base.suffix in {".gz", ".bz2", ".xz"}:
        base = base.with_suffix("")
    if base.suffix == ".tar":
        base = base.with_suffix("")
    report_dir = base.parent / f"{base.name}_report"

    return AnalyzeResponse(
        status="done",
        report_dir=str(report_dir),
        message="Report generated successfully.",
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Получение статуса задачи по её идентификатору.
    """
    output_root = Path("/data").resolve()
    status_file = output_root / "status" / f"{job_id}.json"

    if status_file.is_file():
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            return JobStatusResponse(**data)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=500, detail="Не удалось прочитать статус задачи.")

    # Пытаемся угадать статус по наличию архивов, если статус‑файл не найден
    archives_dir = output_root / "archives"
    archive_opt = archives_dir / f"{job_id}.opt.tar.gz"
    archive_md = archives_dir / f"{job_id}.md.tar.gz"

    if archive_opt.is_file():
        return JobStatusResponse(
            status="done",
            job_id=job_id,
            mode="OPT",
            archive_path=str(archive_opt),
            message="Задача завершена, найден архив OPT, но статус‑файл отсутствует.",
        )
    if archive_md.is_file():
        return JobStatusResponse(
            status="done",
            job_id=job_id,
            mode="MD",
            archive_path=str(archive_md),
            message="Задача завершена, найден архив MD, но статус‑файл отсутствует.",
        )

    raise HTTPException(status_code=404, detail="Статус задачи не найден.")


