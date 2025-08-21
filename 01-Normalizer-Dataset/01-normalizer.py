#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normaliza las etiquetas Dewey de un TXT con formato:
titulo | dewey | descripcion

- Filtra entradas con "sin dewey"
- Convierte códigos Dewey con decimales (ej: 530.12) a la centena (ej: 530)
- Guarda el resultado en un archivo nuevo

Uso:
    python normalizer.py --input ./libros.txt --output ./libros_normalizados.txt
"""

import re
import argparse

def normalize_dewey(label: str) -> str:
    """
    Normaliza el código Dewey:
    - Extrae los primeros 3 dígitos
    - Si no hay match, devuelve la etiqueta original limpia
    """
    label = label.strip().lower()
    if label == "sin dewey" or not label:
        return None
    m = re.search(r"\d{3}", label)
    return m.group(0) if m else label

def process_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        kept, skipped = 0, 0
        for line in fin:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 2:
                skipped += 1
                continue
            title, dewey = parts[0], parts[1]
            desc = parts[2] if len(parts) > 2 else ""
            norm = normalize_dewey(dewey)
            if norm is None:
                skipped += 1
                continue
            fout.write(f"{title} | {norm} | {desc}\n")
            kept += 1
    print(f"✅ Archivo procesado: {kept} líneas guardadas, {skipped} descartadas")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Ruta al TXT original")
    parser.add_argument("--output", type=str, required=True, help="Ruta al TXT normalizado")
    args = parser.parse_args()
    process_file(args.input, args.output)

if __name__ == "__main__":
    main()
