#!/usr/bin/env python3
import argparse
from pathlib import Path

BOHR_TO_ANG = 0.529177210903  # достаточно точно

def read_coord(path):
    """
    Читает md/coord:
    - типы атомов (последний столбец)
    - векторы решётки в борах -> в ангстремы
    """
    atoms = []
    lattice = []

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    mode = "coord"
    for line in lines:
        if line.startswith("$periodic"):
            mode = "periodic"
            continue
        if line.startswith("$lattice"):
            mode = "lattice"
            continue
        if line.startswith("$end"):
            mode = "other"
            continue

        if mode == "coord":
            if line.startswith("$coord"):
                continue
            parts = line.split()
            if len(parts) == 4:
                # x y z elem
                elem = parts[3].strip().upper()
                atoms.append(elem)
        elif mode == "lattice":
            parts = line.split()
            if len(parts) == 3:
                vec = [float(x) * BOHR_TO_ANG for x in parts]
                lattice.append(vec)

    if len(lattice) != 3:
        raise ValueError("Не удалось прочитать 3 вектора решётки из coord")

    return atoms, lattice


def parse_xyz_frames(path):
    """
    Читает многофреймовый XYZ.
    Возвращает список кадров: [(comment, [(elem, x, y, z), ...]), ...]
    """
    frames = []
    with open(path, "r") as f:
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
            atoms = []
            for _ in range(nat):
                l = f.readline()
                if not l:
                    raise ValueError("Неожиданный конец файла XYZ при чтении координат")
                parts = l.split()
                if len(parts) < 4:
                    raise ValueError(f"Неверная строка XYZ: {l!r}")
                elem = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append((elem, x, y, z))
            frames.append((comment, atoms))
    return frames


def cart_to_frac(x, y, z, lattice):
    """
    Перевод декартовых координат (Å) в дробные
    r = x*a1 + y*a2 + z*a3, но здесь подразумевается
    что xyz уже в лабораторных осях, а решётка ортогональная.
    Для общего случая надо решать A * frac = r.
    Здесь используем общий случай через обратную матрицу.
    """
    import numpy as np

    A = np.array(lattice).T  # столбцы – векторы a, b, c
    r = np.array([x, y, z])
    frac = np.linalg.solve(A, r)
    return frac.tolist()


def write_cif(output, frame_index, comment, atoms_in_frame, types_from_coord, lattice):
    """
    Пишет один data_блок CIF для одного кадра.
    """
    # проверяем совпадение числа атомов
    if len(atoms_in_frame) != len(types_from_coord):
        raise ValueError(
            f"В кадре {frame_index} число атомов в XYZ ({len(atoms_in_frame)}) "
            f"не совпадает с coord ({len(types_from_coord)})"
        )

    a1, a2, a3 = lattice
    # длины
    import math
    def norm(v):
        return math.sqrt(sum(vi*vi for vi in v))
    def angle(u, v):
        from math import acos, degrees
        nu = norm(u)
        nv = norm(v)
        return degrees(acos(sum(ui*vi for ui,vi in zip(u,v))/(nu*nv)))

    length_a = norm(a1)
    length_b = norm(a2)
    length_c = norm(a3)
    alpha = angle(a2, a3)
    beta  = angle(a1, a3)
    gamma = angle(a1, a2)

    # отдельный data_блок на каждый кадр
    block_name = f"data_frame_{frame_index:04d}"
    output.write(f"{block_name}\n")
    if comment:
        output.write(f"# {comment}\n")
    output.write("_symmetry_space_group_name_H-M    'P 1'\n")
    output.write("_symmetry_Int_Tables_number       1\n\n")

    output.write(f"_cell_length_a    {length_a:.6f}\n")
    output.write(f"_cell_length_b    {length_b:.6f}\n")
    output.write(f"_cell_length_c    {length_c:.6f}\n")
    output.write(f"_cell_angle_alpha {alpha:.4f}\n")
    output.write(f"_cell_angle_beta  {beta:.4f}\n")
    output.write(f"_cell_angle_gamma {gamma:.4f}\n\n")

    output.write("loop_\n")
    output.write("  _atom_site_label\n")
    output.write("  _atom_site_type_symbol\n")
    output.write("  _atom_site_fract_x\n")
    output.write("  _atom_site_fract_y\n")
    output.write("  _atom_site_fract_z\n")

    import numpy as np
    A = np.array(lattice).T
    Ainv = np.linalg.inv(A)

    for i, ((elem_xyz, x, y, z), elem_ref) in enumerate(zip(atoms_in_frame, types_from_coord), start=1):
        # тип берём из coord, но можно сверять
        elem = elem_ref.upper()
        # декартовые → дробные
        r = np.array([x, y, z])
        frac = Ainv @ r
        fx, fy, fz = frac.tolist()
        label = f"{elem}{i}"
        output.write(f"  {label:<4} {elem:<2} {fx: .6f} {fy: .6f} {fz: .6f}\n")

    output.write("\n")


def main():
    ap = argparse.ArgumentParser(
        description="Конвертер XYZ-траектории в CIF с ячейкой из md/coord"
    )
    ap.add_argument("--coord", required=True, help="файл coord (например md/coord)")
    ap.add_argument("--xyz", required=True, help="XYZ-траектория (с кадрами)")
    ap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="номер кадра (с 1). Если не задан, выводятся все кадры подряд."
    )
    ap.add_argument(
        "-o", "--output",
        default="-",
        help="выходной CIF (по умолчанию stdout)"
    )
    args = ap.parse_args()

    types, lattice = read_coord(args.coord)
    frames = parse_xyz_frames(args.xyz)

    if args.frame is not None:
        idx = args.frame - 1
        if not (0 <= idx < len(frames)):
            raise SystemExit(f"Кадр {args.frame} вне диапазона (1..{len(frames)})")
        out = (open(args.output, "w") if args.output != "-" else None)
        fh = out if out is not None else __import__("sys").stdout
        comment, atoms_frame = frames[idx]
        write_cif(fh, args.frame, comment, atoms_frame, types, lattice)
        if out is not None:
            out.close()
    else:
        out = (open(args.output, "w") if args.output != "-" else None)
        fh = out if out is not None else __import__("sys").stdout
        for i, (comment, atoms_frame) in enumerate(frames, start=1):
            write_cif(fh, i, comment, atoms_frame, types, lattice)
        if out is not None:
            out.close()


if __name__ == "__main__":
    main()