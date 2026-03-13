#!/usr/bin/env python3
import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Molecule:
  name: str
  symbols: List[str]
  coords: np.ndarray


def read_xyz(path: Path, name: str | None = None) -> Molecule:
  with path.open() as f:
    lines = [l.rstrip("\n") for l in f]

  if len(lines) < 2:
    raise ValueError("Некорректный XYZ: слишком мало строк.")
  n_atoms = int(lines[0].strip())
  comment = lines[1].strip() if len(lines) > 1 else (name or path.stem)

  symbols: List[str] = []
  coords = []
  idx = 2
  while len(symbols) < n_atoms:
    if idx >= len(lines):
      raise ValueError(
        f"В {path}: в заголовке указано {n_atoms} атомов, но в файле только {len(symbols)} строк с координатами. "
        f"Проверьте, что после второй строки идёт ровно {n_atoms} строк вида «символ x y z»."
      )
    line = lines[idx].strip()
    idx += 1
    if not line:
      continue
    parts = line.split()
    if len(parts) < 4:
      raise ValueError(f"Некорректная строка атома в XYZ: {line!r}")
    symbols.append(parts[0])
    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

  if len(symbols) != n_atoms:
    raise ValueError(
      f"В {path}: прочитано {len(symbols)} атомов, в заголовке указано {n_atoms}."
    )

  arr = np.asarray(coords, dtype=float)
  center = arr.mean(axis=0)
  arr -= center
  mol_name = comment or (name or path.stem)
  logger.debug(
    "read_xyz: %s, name=%r, n_atoms=%d, symbols_counts=%s",
    path,
    mol_name,
    n_atoms,
    {s: symbols.count(s) for s in sorted(set(symbols))},
  )
  return Molecule(name=mol_name, symbols=symbols, coords=arr)


def write_xyz(path: Path, symbols: List[str], coords: np.ndarray, comment: str = "") -> None:
  n_atoms = len(symbols)
  with path.open("w") as f:
    f.write(f"{n_atoms}\n")
    f.write(f"{comment}\n")
    for sym, (x, y, z) in zip(symbols, coords):
      f.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def random_rotation_matrix() -> np.ndarray:
  u1, u2, u3 = random.random(), random.random(), random.random()
  q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
  q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
  q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
  q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
  q = np.array([q1, q2, q3, q4])
  q1, q2, q3, q4 = q
  return np.array(
    [
      [
        1 - 2 * (q2**2 + q3**2),
        2 * (q1 * q2 - q3 * q4),
        2 * (q1 * q3 + q2 * q4),
      ],
      [
        2 * (q1 * q2 + q3 * q4),
        1 - 2 * (q1**2 + q3**2),
        2 * (q2 * q3 - q1 * q4),
      ],
      [
        2 * (q1 * q3 - q2 * q4),
        2 * (q2 * q3 + q1 * q4),
        1 - 2 * (q1**2 + q2**2),
      ],
    ]
  )


def pack_system(
  mol_defs: List[Tuple[Molecule, int]],
  box_size: float,
  min_dist: float,
  max_attempts_per_mol: int = 5000,
) -> Tuple[List[str], np.ndarray]:
  logger.info(
    "pack_system: старт упаковки, box_size=%.3f Å, min_dist=%.3f Å, max_attempts_per_mol=%d",
    box_size,
    min_dist,
    max_attempts_per_mol,
  )
  for mol, count in mol_defs:
    logger.info(
      "pack_system: компонент name=%r, atoms=%d, count=%d, symbols_counts=%s",
      mol.name,
      len(mol.symbols),
      count,
      {s: mol.symbols.count(s) for s in sorted(set(mol.symbols))},
    )
  all_symbols: List[str] = []
  all_coords_list: List[np.ndarray] = []

  existing_coords = np.empty((0, 3), dtype=float)
  min_dist_sq = float(min_dist * min_dist)

  # формируем последовательность молекул разных типов по очереди
  remaining = [count for _, count in mol_defs]
  sequence: List[Molecule] = []
  while any(c > 0 for c in remaining):
    for idx, (mol, _) in enumerate(mol_defs):
      if remaining[idx] > 0:
        sequence.append(mol)
        remaining[idx] -= 1

  logger.debug(
    "pack_system: длина итоговой последовательности молекул для размещения: %d",
    len(sequence),
  )

  for mol_idx, mol in enumerate(sequence, start=1):
      logger.debug(
        "pack_system: размещение молекулы #%d name=%r (atoms=%d)",
        mol_idx,
        mol.name,
        len(mol.symbols),
      )
      placed = False
      for _attempt in range(max_attempts_per_mol):
        R = random_rotation_matrix()
        rotated = mol.coords @ R.T

        shift = np.array(
          [
            random.uniform(0.0, box_size),
            random.uniform(0.0, box_size),
            random.uniform(0.0, box_size),
          ]
        )
        new_coords = rotated + shift

        if existing_coords.size == 0:
          placed = True
          break

        diff = new_coords[:, None, :] - existing_coords[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_d2 = float(d2.min())
        if min_d2 >= min_dist_sq:
          placed = True
          break
      logger.debug(
        "pack_system: попыток для молекулы #%d name=%r: %d, min_dist^2=%.4f",
        mol_idx,
        mol.name,
        _attempt + 1,
        min_dist_sq,
      )

      if not placed:
        raise RuntimeError(
          f"Не удалось разместить молекулу {mol.name} "
          f"с min_dist={min_dist} Å в коробке {box_size} Å. "
          f"Увеличьте box_size или уменьшите min_dist."
        )

      all_symbols.extend(mol.symbols)
      all_coords_list.append(new_coords)
      existing_coords = np.vstack([existing_coords, new_coords])

  all_coords = np.vstack(all_coords_list)
  return all_symbols, all_coords


def estimate_min_box_size(
  mol_defs: List[Tuple[Molecule, int]],
  min_dist: float,
  packing_fraction: float = 0.7,
) -> float:
  logger.info(
    "estimate_min_box_size: min_dist=%.3f Å, packing_fraction=%.3f",
    min_dist,
    packing_fraction,
  )
  total_vol = 0.0
  extra = 0.5 * float(min_dist)

  for mol, count in mol_defs:
    if mol.coords.size == 0 or count <= 0:
      continue
    r = float(np.linalg.norm(mol.coords, axis=1).max()) + extra
    vol = (4.0 / 3.0) * math.pi * (r**3)
    total_vol += vol * max(count, 0)
    logger.debug(
      "estimate_min_box_size: mol=%r, atoms=%d, count=%d, r=%.3f, vol_one=%.3f",
      mol.name,
      len(mol.symbols),
      count,
      r,
      vol,
    )

  if total_vol <= 0.0:
    logger.warning(
      "estimate_min_box_size: total_vol<=0, возвращаю дефолтный размер коробки 10.0 Å"
    )
    return 10.0

  box = (total_vol / max(packing_fraction, 1e-3)) ** (1.0 / 3.0)
  box_clamped = max(float(box), 5.0)
  logger.info(
    "estimate_min_box_size: total_vol=%.3f, raw_box=%.3f Å, box_clamped=%.3f Å",
    total_vol,
    box,
    box_clamped,
  )
  return box_clamped


def pack_system_auto_box(
  mol_defs: List[Tuple[Molecule, int]],
  min_dist: float,
  max_attempts_per_mol: int = 5000,
  max_box_increase_factor: float = 3.0,
  packing_fraction: float = 1.5,
) -> Tuple[List[str], np.ndarray, float]:
  logger.info(
    "pack_system_auto_box: старт, min_dist=%.3f Å, max_attempts_per_mol=%d, max_box_increase_factor=%.2f",
    min_dist,
    max_attempts_per_mol,
    max_box_increase_factor,
  )
  base_box = estimate_min_box_size(mol_defs, min_dist, packing_fraction=packing_fraction)
  box = base_box
  max_box = base_box * max_box_increase_factor

  while box <= max_box:
    try:
      logger.info(
        "pack_system_auto_box: попытка упаковки с box=%.3f Å (max_box=%.3f Å)",
        box,
        max_box,
      )
      symbols, coords = pack_system(
        mol_defs,
        box_size=box,
        min_dist=min_dist,
        max_attempts_per_mol=max_attempts_per_mol,
      )
      logger.info(
        "pack_system_auto_box: успешная упаковка при box=%.3f Å, всего атомов=%d",
        box,
        len(symbols),
      )
      return symbols, coords, box
    except RuntimeError:
      logger.warning(
        "pack_system_auto_box: не удалось упаковать при box=%.3f Å, увеличиваю коробку на 10%%",
        box,
      )
      box *= 1.10

  raise RuntimeError(
    "Не удалось упаковать систему с автоматическим подбором объёма. "
    "Попробуйте уменьшить min_dist или сократить количество молекул."
  )


def parse_mol_spec(spec: str) -> Tuple[Path, int]:
  if ":" not in spec:
    raise ValueError(f"Ожидался формат path.xyz:count, получено: {spec}")
  path_str, count_str = spec.split(":", 1)
  return Path(path_str), int(count_str)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Сборка многокомпонентной системы из XYZ с контролем минимального расстояния."
  )
  parser.add_argument(
    "-m",
    "--molecule",
    action="append",
    required=True,
    help="Молекула в формате path.xyz:count (можно несколько раз).",
  )
  parser.add_argument(
    "--min-dist",
    type=float,
    default=2.0,
    help="Минимальное расстояние между атомами (Å). По умолчанию 2.0.",
  )
  parser.add_argument(
    "--packing-fraction",
    type=float,
    default=0.7,
    help="Целевая эффективная доля заполнения объёма (0–1). "
    "Большие значения дают более плотную упаковку (меньше коробка). "
    "По умолчанию 0.7.",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=Path("system.xyz"),
    help="Имя выходного XYZ‑файла.",
  )

  args = parser.parse_args()

  mol_defs: List[Tuple[Molecule, int]] = []
  for spec in args.molecule:
    path, count = parse_mol_spec(spec)
    mol = read_xyz(path)
    mol_defs.append((mol, count))

  symbols, coords, box_size = pack_system_auto_box(
    mol_defs,
    min_dist=args.min_dist,
    packing_fraction=args.packing_fraction,
  )
  comment = f"Packed system; auto box={box_size:.3f} Å; min_dist={args.min_dist} Å"
  write_xyz(args.output, symbols, coords, comment=comment)


if __name__ == "__main__":
  main()

