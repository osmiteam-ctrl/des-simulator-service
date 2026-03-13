#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Использование: $0 input.smi output_dir" >&2
  echo "Формат input.smi: <smiles> <name> на строку" >&2
  exit 1
fi

INPUT_SMI="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

i=1
while read -r line; do
  # пропускаем пустые строки и комментарии
  [ -z "$line" ] && continue
  [[ "$line" =~ ^# ]] && continue

  smiles=$(echo "$line" | awk '{print $1}')
  name=$(echo "$line" | awk '{print $2}')
  if [ -z "${name:-}" ]; then
    name="mol_$i"
  fi

  obabel -:"$smiles" -oxyz --gen3d -O "$OUTPUT_DIR/${name}.xyz"
  i=$((i+1))
done < "$INPUT_SMI"

