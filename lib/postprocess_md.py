#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# Обеспечиваем импорт lib.* при запуске как скрипта: python lib/postprocess_md.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from lib.xyz2cif import read_coord as read_tm_coord  # для чтения ячейки из coord


def parse_xtb_trj(path: Path, dump_fs: float) -> Tuple[List[dict[str, Any]], List[np.ndarray], List[List[str]]]:
  """
  Парсит xtb.trj:
  - первая строка кадра: число атомов N
  - вторая строка: 'energy: ... gnorm: ... ...'
  - далее N строк: elem x y z

  Возвращает:
    frames_meta: список словарей с time_ps, energy, gnorm
    coords_list: список (N,3) массивов координат
    symbols_list: список списков типов атомов
  """
  frames_meta: List[dict[str, Any]] = []
  coords_list: List[np.ndarray] = []
  symbols_list: List[List[str]] = []

  dump_ps = float(dump_fs) * 1e-3  # fs -> ps
  frame_idx = 0

  with path.open() as f:
    while True:
      line = f.readline()
      if not line:
        break
      line = line.strip()
      if not line:
        continue
      try:
        n_atoms = int(line)
      except ValueError:
        # неожиданная строка – формат не тот
        raise ValueError(f"Ожидалось число атомов в xtb.trj, но строка: {line!r}")

      header = f.readline()
      if not header:
        break
      header = header.strip()

      energy = math.nan
      gnorm = math.nan
      parts = header.replace("=", " ").replace(":", " ").split()
      # ищем по ключевым словам
      for i, p in enumerate(parts):
        if p.lower().startswith("energy"):
          # 'energy', значение рядом
          try:
            energy = float(parts[i + 1])
          except Exception:
            pass
        if p.lower().startswith("gnorm"):
          try:
            gnorm = float(parts[i + 1])
          except Exception:
            pass

      symbols: List[str] = []
      coords = np.zeros((n_atoms, 3), dtype=float)
      for i in range(n_atoms):
        atom_line = f.readline()
        if not atom_line:
          raise ValueError("Неожиданный конец xtb.trj при чтении координат")
        a_parts = atom_line.split()
        if len(a_parts) < 4:
          raise ValueError(f"Некорректная строка атома в xtb.trj: {atom_line!r}")
        sym = a_parts[0].capitalize()
        x, y, z = map(float, a_parts[1:4])
        symbols.append(sym)
        coords[i] = [x, y, z]

      time_ps = frame_idx * dump_ps
      frames_meta.append({"time_ps": time_ps, "energy": energy, "gnorm": gnorm})
      coords_list.append(coords)
      symbols_list.append(symbols)
      frame_idx += 1

  return frames_meta, coords_list, symbols_list


def write_energy_csv(frames_meta: List[dict[str, Any]], path: Path) -> None:
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_ps", "energy", "gnorm"])
    for m in frames_meta:
      writer.writerow([m["time_ps"], m["energy"], m["gnorm"]])


def compute_hbonds_for_frame(coords: np.ndarray, symbols: List[str]) -> Tuple[int, float, float]:
  """
  Простейшая оценка водородных связей:
  - определяем H, O, N
  - для каждого H находим ближайший донор (O/N) в радиусе 1.2 Å
  - для каждой пары донор-акцептор (O/N != донор):
      r_DA < 3.5 Å и угол D-H-A > 150°
  Возвращает:
    count, avg_distance_DA, avg_angle_DHA
  """
  coords = np.asarray(coords, dtype=float)
  n = coords.shape[0]
  idx_H = [i for i, s in enumerate(symbols) if s == "H"]
  idx_X = [i for i, s in enumerate(symbols) if s in {"O", "N"}]

  if not idx_H or len(idx_X) < 2:
    return 0, math.nan, math.nan

  hb_dist: List[float] = []
  hb_angle: List[float] = []

  X_coords = coords[idx_X]  # (NX,3)

  for ih in idx_H:
    H = coords[ih]
    # расстояния до потенциальных доноров O/N
    diff = X_coords - H
    d2 = np.sum(diff * diff, axis=1)
    j_min = int(np.argmin(d2))
    d_min = math.sqrt(float(d2[j_min]))
    if d_min > 1.2:
      continue  # не нашли связанный донор

    donor_idx = idx_X[j_min]
    D = coords[donor_idx]

    # кандидаты в акцепторы
    for ja, a_idx in enumerate(idx_X):
      if a_idx == donor_idx:
        continue
      A = coords[a_idx]
      DA_vec = A - D
      r_DA = float(np.linalg.norm(DA_vec))
      if r_DA > 3.5:
        continue

      # угол D-H-A
      vDH = H - D
      vHA = A - H
      n1 = np.linalg.norm(vDH)
      n2 = np.linalg.norm(vHA)
      if n1 < 1e-6 or n2 < 1e-6:
        continue
      cosang = float(np.dot(vDH, vHA) / (n1 * n2))
      cosang = max(-1.0, min(1.0, cosang))
      angle = math.degrees(math.acos(cosang))
      if angle >= 150.0:
        hb_dist.append(r_DA)
        hb_angle.append(angle)

  if not hb_dist:
    return 0, math.nan, math.nan

  count = len(hb_dist)
  avg_d = float(sum(hb_dist) / count)
  avg_a = float(sum(hb_angle) / count)
  return count, avg_d, avg_a


def write_hbonds_csv(
  frames_meta: List[dict[str, Any]],
  coords_list: List[np.ndarray],
  symbols_list: List[List[str]],
  path: Path,
) -> None:
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_ps", "n_hbonds", "avg_distance", "avg_angle"])
    for meta, coords, symbols in zip(frames_meta, coords_list, symbols_list):
      n_hb, avg_d, avg_a = compute_hbonds_for_frame(coords, symbols)
      writer.writerow([meta["time_ps"], n_hb, avg_d, avg_a])


def min_image(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
  return dr - box * np.round(dr / box)


def compute_rdf_pair(
  coords_list: List[np.ndarray],
  symbols_list: List[List[str]],
  type1: str,
  type2: str,
  r_max: float,
  dr: float,
  box: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
  type1 = type1.capitalize()
  type2 = type2.capitalize()

  edges = np.arange(0.0, r_max + dr, dr)
  centers = 0.5 * (edges[:-1] + edges[1:])
  hist = np.zeros_like(centers)

  n_frames_used = 0
  vol = None

  for coords, symbols in zip(coords_list, symbols_list):
    coords = np.asarray(coords, dtype=float)
    symbols = list(symbols)
    if box is not None:
      if vol is None:
        vol = float(box[0] * box[1] * box[2])
    else:
      mins = coords.min(axis=0)
      maxs = coords.max(axis=0)
      lengths = maxs - mins
      if vol is None:
        vol = float(lengths[0] * lengths[1] * lengths[2])

    idx1 = np.array([i for i, s in enumerate(symbols) if s == type1])
    idx2 = np.array([i for i, s in enumerate(symbols) if s == type2])
    if len(idx1) == 0 or len(idx2) == 0:
      continue

    pos1 = coords[idx1]
    pos2 = coords[idx2]

    dr_vec = pos1[:, None, :] - pos2[None, :, :]
    if box is not None:
      dr_vec = min_image(dr_vec, box)

    dist = np.linalg.norm(dr_vec, axis=2).ravel()
    if type1 == type2:
      dist = dist[dist > 1e-12]

    h, _ = np.histogram(dist, bins=edges)
    hist += h
    n_frames_used += 1

  if n_frames_used == 0 or vol is None:
    raise RuntimeError(f"Не удалось посчитать RDF для пары {type1}-{type2}")

  # нормировка
  for coords, symbols in zip(coords_list, symbols_list):
    coords = np.asarray(coords, dtype=float)
    symbols = list(symbols)
    idx1 = [i for i, s in enumerate(symbols) if s == type1]
    idx2 = [i for i, s in enumerate(symbols) if s == type2]
    if idx1 and idx2:
      n1 = len(idx1)
      n2 = len(idx2)
      break
  else:
    raise RuntimeError(f"Не удалось определить количество атомов для {type1}-{type2}")

  rho2 = n2 / vol
  shell_vol = 4.0 * np.pi * centers**2 * dr
  ideal_counts = n_frames_used * n1 * rho2 * shell_vol

  g_r = hist / np.maximum(ideal_counts, 1e-12)
  return centers, g_r


def write_rdf_csv(
  coords_list: List[np.ndarray],
  symbols_list: List[List[str]],
  box: np.ndarray | None,
  path: Path,
  r_max: float = 10.0,
  dr: float = 0.1,
) -> None:
  pairs = [("O", "H"), ("O", "O"), ("N", "H"), ("N", "O"), ("N", "N")]
  r = None
  columns: dict[str, np.ndarray] = {}

  for a, b in pairs:
    r_tmp, g = compute_rdf_pair(coords_list, symbols_list, a, b, r_max, dr, box)
    if r is None:
      r = r_tmp
    columns[f"g_{a}{b}"] = g

  assert r is not None

  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    header = ["r"] + [f"g_{a}{b}" for a, b in pairs]
    writer.writerow(header)
    for i in range(len(r)):
      row = [r[i]] + [columns[f"g_{a}{b}"][i] for a, b in pairs]
      writer.writerow(row)


def get_box_from_coord(coord_path: Path) -> np.ndarray | None:
  """
  Читает ячейку из Turbomole coord (через read_tm_coord из xyz2cif).
  Возвращает вектор (Lx, Ly, Lz) в ангстремах, если удалось.
  """
  try:
    _, lattice = read_tm_coord(str(coord_path))
  except Exception:
    return None
  if len(lattice) != 3:
    return None
  a1, a2, a3 = lattice
  return np.array(
    [
      float(np.linalg.norm(a1)),
      float(np.linalg.norm(a2)),
      float(np.linalg.norm(a3)),
    ],
    dtype=float,
  )


def write_system_params_csv(frames_meta: List[dict[str, Any]], path: Path) -> None:
  """
  Пока просто дублируем время, энергию и gnorm как базовый набор параметров.
  При желании можно позже расширить (температура, давление и т.п., если появятся источники).
  """
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_ps", "energy", "gnorm"])
    for m in frames_meta:
      writer.writerow([m["time_ps"], m["energy"], m["gnorm"]])


def main() -> None:
  ap = argparse.ArgumentParser(
    description="Постобработка результатов MD (xtb.trj + coord): энергия, H‑связи, RDF, системные параметры."
  )
  ap.add_argument("--trj", type=Path, required=True, help="Файл xtb.trj.")
  ap.add_argument("--coord", type=Path, required=True, help="Файл coord (Turbomole).")
  ap.add_argument(
    "--dump-fs",
    type=float,
    required=True,
    help="Период записи кадров (dump) в фемтосекундах.",
  )
  ap.add_argument(
    "--out-dir",
    type=Path,
    required=True,
    help="Каталог, куда писать CSV (energy.csv, hbonds.csv, rdf.csv, system_params.csv).",
  )
  ap.add_argument(
    "--r-max",
    type=float,
    default=10.0,
    help="Максимальный радиус для RDF (Å). По умолчанию 10.0.",
  )
  ap.add_argument(
    "--dr",
    type=float,
    default=0.1,
    help="Шаг по r для RDF (Å). По умолчанию 0.1.",
  )

  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  frames_meta, coords_list, symbols_list = parse_xtb_trj(args.trj, args.dump_fs)

  energy_csv = args.out_dir / "energy.csv"
  hbonds_csv = args.out_dir / "hbonds.csv"
  rdf_csv = args.out_dir / "rdf.csv"
  sys_csv = args.out_dir / "system_params.csv"

  write_energy_csv(frames_meta, energy_csv)
  write_hbonds_csv(frames_meta, coords_list, symbols_list, hbonds_csv)

  box = get_box_from_coord(args.coord)
  write_rdf_csv(coords_list, symbols_list, box, rdf_csv, r_max=args.r_max, dr=args.dr)

  write_system_params_csv(frames_meta, sys_csv)


if __name__ == "__main__":
  main()

