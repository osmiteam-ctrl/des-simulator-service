#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_xyz_frames(path: Path) -> Tuple[List[str], List[np.ndarray], List[List[str]]]:
    """
    Читает многофреймовый XYZ.
    Возвращает:
      comments  – список комментариев по кадрам (можно игнорировать)
      coords    – список массивов (n_atoms, 3) для каждого кадра
      symbols   – список списков типов атомов для каждого кадра
    """
    comments: List[str] = []
    all_coords: List[np.ndarray] = []
    all_symbols: List[List[str]] = []

    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            nat_line = line.strip()
            if not nat_line:
                continue
            try:
                nat = int(nat_line)
            except ValueError:
                raise ValueError(f"Ожидалось число атомов, но строка: {nat_line!r}")
            comment = f.readline().rstrip("\n")
            symbols: List[str] = []
            coords = np.zeros((nat, 3), dtype=float)
            for i in range(nat):
                l = f.readline()
                if not l:
                    raise ValueError("Неожиданный конец файла XYZ")
                parts = l.split()
                if len(parts) < 4:
                    raise ValueError(f"Некорректная строка XYZ: {l!r}")
                symbols.append(parts[0])
                coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
            comments.append(comment)
            all_coords.append(coords)
            all_symbols.append(symbols)

    return comments, all_coords, all_symbols


def min_image(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Применяет minimum-image convention по ортогональной ячейке.
    dr: (..., 3), box: (3,) (Lx, Ly, Lz)
    """
    return dr - box * np.round(dr / box)


def compute_rdf(
    coords_list: List[np.ndarray],
    symbols_list: List[List[str]],
    type1: str,
    type2: str,
    r_max: float,
    dr: float,
    box: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Считает g(r) между типами атомов type1 и type2.
    coords_list: список координат по кадрам (n_atoms, 3)
    symbols_list: список списков символов по кадрам
    box: (3,) – длины ячейки (Å) или None (без PBC)
    """
    type1 = type1.capitalize()
    type2 = type2.capitalize()

    # радиальные бины
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
            # объём оценим по габаритам этого кадра (очень грубо)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            lengths = maxs - mins
            if vol is None:
                vol = float(lengths[0] * lengths[1] * lengths[2])

        idx1 = np.array([i for i, s in enumerate(symbols) if s == type1])
        idx2 = np.array([i for i, s in enumerate(symbols) if s == type2])
        if len(idx1) == 0 or len(idx2) == 0:
            continue  # в этом кадре нет нужных типов

        pos1 = coords[idx1]  # (N1, 3)
        pos2 = coords[idx2]  # (N2, 3)

        # всевозможные пары
        dr_vec = pos1[:, None, :] - pos2[None, :, :]  # (N1, N2, 3)
        if box is not None:
            dr_vec = min_image(dr_vec, box)

        dist = np.linalg.norm(dr_vec, axis=2).ravel()

        # если type1 == type2 – убрать пары i=i
        if type1 == type2:
            # диагональ расстояний = 0, просто отфильтруем
            dist = dist[dist > 1e-12]

        # обновляем гистограмму
        h, _ = np.histogram(dist, bins=edges)
        hist += h
        n_frames_used += 1

    if n_frames_used == 0 or vol is None:
        raise RuntimeError("Не удалось найти подходящие атомы или объём системы.")

    # нормировка
    # число атомов каждого типа (по первому кадру с нужными типами)
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
        raise RuntimeError("Не удалось определить количество атомов нужных типов.")

    rho2 = n2 / vol  # плотность второго типа (1/Å^3)

    shell_vol = 4.0 * np.pi * centers**2 * dr  # объём сферической оболочки
    ideal_counts = n_frames_used * n1 * rho2 * shell_vol

    g_r = hist / np.maximum(ideal_counts, 1e-12)

    return centers, g_r


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Радиальное распределение g(r) по XYZ‑траектории."
    )
    ap.add_argument("--xyz", required=True, type=Path, help="XYZ с кадрами MD.")
    ap.add_argument(
        "--type1", required=True, help="Первый тип атомов (например, O, N, C)."
    )
    ap.add_argument(
        "--type2", required=True, help="Второй тип атомов (например, H, O, C)."
    )
    ap.add_argument(
        "--r-max", type=float, default=10.0, help="Максимальный радиус (Å)."
    )
    ap.add_argument("--dr", type=float, default=0.1, help="Шаг по r (Å).")
    ap.add_argument(
        "--box",
        type=float,
        nargs="+",
        help="Длины ячейки (Å): либо одна величина L (куб), либо три Lx Ly Lz. "
             "Если не задано — PBC не используются, объём оценивается по габаритам.",
    )
    ap.add_argument(
        "--show-only",
        action="store_true",
        help="Игнорировать вывод в файл и просто показать график (по умолчанию так и делается).",
    )

    args = ap.parse_args()

    _, coords_list, symbols_list = parse_xyz_frames(args.xyz)

    box = None
    if args.box is not None:
        if len(args.box) == 1:
            L = float(args.box[0])
            box = np.array([L, L, L], dtype=float)
        elif len(args.box) == 3:
            box = np.array(args.box, dtype=float)
        else:
            raise SystemExit("Аргумент --box: нужно либо 1 число (L), либо 3 (Lx Ly Lz).")

    r, g_r = compute_rdf(
        coords_list,
        symbols_list,
        type1=args.type1,
        type2=args.type2,
        r_max=args.r_max,
        dr=args.dr,
        box=box,
    )

    # Рисуем g(r) через matplotlib
    plt.figure(figsize=(6, 4))
    plt.plot(r, g_r, "-", lw=1.5)
    plt.xlabel(r"r, $\mathrm{\AA}$")
    plt.ylabel("g(r)")
    plt.title(f"RDF {args.type1}-{args.type2}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()