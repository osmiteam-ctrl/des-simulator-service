#!/usr/bin/env python3
import argparse
import json
import logging
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pack_xyz import (
  Molecule,
  pack_system_auto_box,
  read_xyz,
  write_xyz,
)

logger = logging.getLogger(__name__)


def generate_xyz_with_obabel(smiles: str, name: str, tmp_dir: Path) -> Path:
  out_path = tmp_dir / f"{name}.xyz"
  cmd = [
    "obabel",
    f"-:{smiles}",
    "-oxyz",
    "-h",
    "--gen3d",
    "-O",
    str(out_path),
  ]
  logger.debug("obabel: %s", " ".join(cmd))
  try:
    result = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      check=True,
    )
    if result.stderr and logger.isEnabledFor(logging.DEBUG):
      logger.debug("obabel stderr: %s", result.stderr.strip())
  except subprocess.CalledProcessError as e:
    logger.error(
      "obabel failed для SMILES %r (name=%s): exitcode=%s",
      smiles,
      name,
      e.returncode,
    )
    if e.stdout:
      logger.error("obabel stdout: %s", e.stdout.strip())
    if e.stderr:
      logger.error("obabel stderr: %s", e.stderr.strip())
    raise
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Построение многокомпонентной системы из JSON описания (SMILES) в XYZ."
  )
  parser.add_argument(
    "json_file",
    type=Path,
    help="JSON‑файл с массивом объектов {name, smiles, count}.",
  )
  parser.add_argument(
    "--min-dist",
    type=float,
    default=2.0,
    help="Минимальное расстояние между атомами (Å). По умолчанию 2.0.",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=Path("system.xyz"),
    help="Имя выходного XYZ‑файла.",
  )
  parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Подробный вывод (DEBUG).",
  )

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(levelname)s: %(message)s",
  )

  with args.json_file.open() as f:
    data = json.load(f)

  if not isinstance(data, list):
    raise ValueError("JSON должен содержать список объектов.")

  logger.info("Компонентов в JSON: %d", len(data))

  with tempfile.TemporaryDirectory() as tmpdir_str:
    tmpdir = Path(tmpdir_str)
    logger.debug("Временный каталог: %s", tmpdir)

    mol_defs: List[Tuple[Molecule, int]] = []
    for i, item in enumerate(data):
      name = str(item["name"])
      smiles = str(item["smiles"])
      count = int(item["count"])
      logger.info(
        "[%d/%d] name=%r smiles=%r count=%d",
        i + 1,
        len(data),
        name,
        smiles,
        count,
      )

      xyz_path = generate_xyz_with_obabel(smiles, name, tmpdir)
      logger.debug("Создан XYZ: %s", xyz_path)

      mol = read_xyz(xyz_path, name=name)
      n_atoms = len(mol.symbols)
      comp = ", ".join(f"{s}:{c}" for s, c in sorted(Counter(mol.symbols).items()))
      logger.info(
        "  прочитано атомов: %d (%s)",
        n_atoms,
        comp,
      )
      if n_atoms == 0:
        logger.warning("  молекула пустая для name=%r smiles=%r", name, smiles)
      mol_defs.append((mol, count))

    logger.info("Упаковка системы (min_dist=%.2f Å)...", args.min_dist)
    symbols, coords, box_size = pack_system_auto_box(
      mol_defs,
      min_dist=args.min_dist,
    )
    logger.info("Упаковка завершена: box=%.3f Å, всего атомов=%d", box_size, len(symbols))

  comment = (
    f"System from JSON; auto box={box_size:.3f} Å; "
    f"min_dist={args.min_dist} Å; components={len(data)}"
  )
  write_xyz(args.output, symbols, coords, comment=comment)
  logger.info("Записано в %s", args.output)

  # Дополнительно сохраняем маппинг атомов -> молекулы/типы для последующего MSD‑анализа.
  # В pack_system_auto_box молекулы размещаются в порядке "карусели", повторим его здесь.
  mapping_path = args.output.with_suffix(".mapping.json")
  logger.info("Сохраняю атомный маппинг в %s", mapping_path)

  # Восстанавливаем последовательность молекул (в том же порядке, что и в pack_xyz)
  remaining = [count for _, count in mol_defs]
  sequence: List[Molecule] = []
  while any(c > 0 for c in remaining):
    for idx, (mol, _) in enumerate(mol_defs):
      if remaining[idx] > 0:
        sequence.append(mol)
        remaining[idx] -= 1

  # Группируем по типу (name)
  grouped: Dict[str, Dict[str, object]] = defaultdict(lambda: {"name": "", "natoms": 0, "count": 0, "instances": []})
  start = 0
  for mol in sequence:
    name = mol.name
    nat = len(mol.symbols)
    end = start + nat - 1
    g = grouped[name]
    if not g["name"]:
      g["name"] = name
      g["natoms"] = nat
    g["count"] = int(g.get("count", 0)) + 1  # type: ignore[arg-type]
    g["instances"].append({"start": start, "end": end})  # type: ignore[assignment]
    start = end + 1

  mapping_obj = {
    "box_size_A": box_size,
    "components": data,
    "molecules": list(grouped.values()),
  }
  with mapping_path.open("w", encoding="utf-8") as mf:
    json.dump(mapping_obj, mf, indent=2, ensure_ascii=False)
  logger.info("Маппинг записан.")


if __name__ == "__main__":
  main()

