#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

ANG_TO_BOHR = 1.889726124626


def box_from_coords(
  coords: List[Tuple[float, float, float]],
  padding: float = 0.0,
) -> Tuple[float, float, float]:
  """
  Размеры ячейки по каждой координате отдельно.

  Берём габаритный прямоугольник по x, y, z и добавляем padding
  по каждой оси: Lx = (x_max - x_min) + 2*padding и т.д.
  """
  if not coords:
    # минимальная «разумная» ячейка, если координат нет
    base = 2.0 * padding
    return base, base, base

  xs = [c[0] for c in coords]
  ys = [c[1] for c in coords]
  zs = [c[2] for c in coords]

  lx = (max(xs) - min(xs)) + 2.0 * padding
  ly = (max(ys) - min(ys)) + 2.0 * padding
  lz = (max(zs) - min(zs)) + 2.0 * padding

  # на всякий случай не даём длине ячейки быть совсем нулевой/отрицательной
  eps = 1e-3
  lx = max(lx, eps)
  ly = max(ly, eps)
  lz = max(lz, eps)

  return lx, ly, lz


def read_xyz(path: Path) -> Tuple[List[str], List[Tuple[float, float, float]], str]:
  with path.open() as f:
    lines = [l.rstrip("\n") for l in f]

  if len(lines) < 2:
    raise ValueError("Некорректный XYZ: слишком мало строк.")

  try:
    n_atoms = int(lines[0].strip())
  except ValueError as e:
    raise ValueError("Первая строка XYZ должна содержать число атомов.") from e

  comment = lines[1].strip()

  symbols: List[str] = []
  coords: List[Tuple[float, float, float]] = []
  idx = 2
  while len(symbols) < n_atoms:
    if idx >= len(lines):
      raise ValueError(
        f"Некорректный XYZ: ожидалось {n_atoms} атомов, прочитано {len(symbols)}."
      )
    line = lines[idx].strip()
    idx += 1
    if not line:
      continue
    parts = line.split()
    if len(parts) < 4:
      raise ValueError(f"Некорректная строка атома в XYZ: {line!r}")
    sym = parts[0]
    x, y, z = map(float, parts[1:4])
    symbols.append(sym)
    coords.append((x, y, z))

  return symbols, coords, comment


def _wrap_into_cell(
  coords: List[Tuple[float, float, float]],
  cx: float, cy: float, cz: float,
  lx: float, ly: float, lz: float,
) -> List[Tuple[float, float, float]]:
  half = (lx * 0.5, ly * 0.5, lz * 0.5)
  out = []
  for x, y, z in coords:
    nx = (x - cx + half[0]) % lx
    ny = (y - cy + half[1]) % ly
    nz = (z - cz + half[2]) % lz
    if nx < 0:
      nx += lx
    if ny < 0:
      ny += ly
    if nz < 0:
      nz += lz
    out.append((nx, ny, nz))
  return out


def write_coord(
  path: Path,
  symbols: List[str],
  coords: List[Tuple[float, float, float]],
  lx: float,
  ly: float,
  lz: float,
  periodic_dim: int = 3,
  lattice_format: str = "lattice",
  center_in_cell: bool = False,
) -> None:
  if center_in_cell and coords:
    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    coords = _wrap_into_cell(coords, cx, cy, cz, lx, ly, lz)

  with path.open("w") as f:
    f.write("$coord angs\n")
    for sym, (x, y, z) in zip(symbols, coords):
      f.write(f"  {x:16.8f}  {y:16.8f}  {z:16.8f}  {sym.lower()}\n")

    f.write("$periodic " + str(int(periodic_dim)) + "\n")
    if lattice_format == "cell":
      f.write("$cell angs\n")
      f.write(f"  {lx:16.8f}  {ly:16.8f}  {lz:16.8f}  90.00000000  90.00000000  90.00000000\n")
    else:
      ax, ay, az = lx * ANG_TO_BOHR, 0.0, 0.0
      bx, by, bz = 0.0, ly * ANG_TO_BOHR, 0.0
      cx, cy, cz = 0.0, 0.0, lz * ANG_TO_BOHR
      f.write("$lattice bohr\n")
      f.write(f"  {ax:20.14f}  {ay:20.14f}  {az:20.14f}\n")
      f.write(f"  {bx:20.14f}  {by:20.14f}  {bz:20.14f}\n")
      f.write(f"  {cx:20.14f}  {cy:20.14f}  {cz:20.14f}\n")
    f.write("$end\n")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Конвертация XYZ в Turbomole coord с периодическими условиями."
  )
  parser.add_argument("xyz", type=Path, help="Входной XYZ-файл.")
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=Path("coord"),
    help="Имя выходного файла coord.",
  )
  parser.add_argument(
    "--box-size",
    type=float,
    help="Размер кубической ячейки (Å). Если не задан, ячейка считается по модели (центр масс, макс. расстояние + padding).",
  )
  parser.add_argument(
    "--cell",
    nargs=3,
    type=float,
    metavar=("LX", "LY", "LZ"),
    help="Векторы ячейки по осям (Å). Переопределяет --box-size.",
  )
  parser.add_argument(
    "--padding",
    type=float,
    default=2.0,
    help="Добавка к макс. расстоянию от центра при расчёте ячейки по модели (Å). По умолчанию 2.0.",
  )
  parser.add_argument(
    "--periodic-dim",
    type=int,
    default=3,
    help="Размерность периодичности (1, 2 или 3). По умолчанию 3.",
  )
  parser.add_argument(
    "--format",
    choices=("lattice", "cell"),
    default="lattice",
    help="Формат ячейки: lattice ($lattice bohr, три вектора) или cell ($cell angs, шесть чисел, для xtb). По умолчанию lattice.",
  )
  parser.add_argument(
    "--center",
    action="store_true",
    help="Сдвинуть координаты в ячейку [0,L] (по центру масс). Рекомендуется для xtb.",
  )

  args = parser.parse_args()

  symbols, coords, comment = read_xyz(args.xyz)

  if args.cell is not None:
    lx, ly, lz = args.cell
  elif args.box_size is not None:
    lx = ly = lz = args.box_size
  else:
    # Авто‑режим: подбираем ячейку ровно по крайним атомам
    # (без дополнительного padding).
    arr = np.asarray(coords, dtype=float)

    # опционально центрируем по центру масс (влияет только на форму,
    # но не даёт лишнего пустого места, т.к. ячейка всё равно по экстремумам)
    if args.center and arr.size > 0:
      center = arr.mean(axis=0)
      arr -= center

    # длины ячейки по габаритам без запаса
    lx, ly, lz = box_from_coords(arr.tolist(), padding=0.0)

    # сдвиг: делаем так, чтобы min по каждой оси был ровно 0.0
    if arr.size > 0:
      mins = arr.min(axis=0)
      shift = -mins
      arr += shift

    coords = [(float(x), float(y), float(z)) for x, y, z in arr]

  write_coord(
    args.output,
    symbols,
    coords,
    lx=lx,
    ly=ly,
    lz=lz,
    periodic_dim=args.periodic_dim,
    lattice_format=args.format,
    # центрирование и подгонка ячейки уже сделаны выше
    center_in_cell=False,
  )


if __name__ == "__main__":
  main()

