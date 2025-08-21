#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reclasifica libros con códigos Dewey detallados a categorías generalistas (000–900).
Entrada esperada: titulo | dewey | descripcion
Salida: titulo | categoria_general (ej: 600) | nombre_categoria | descripcion
"""

import os

# Mapa Dewey generalista
DEWEY_GENERAL = {
    "0": "Generalidades",
    "1": "Filosofía y Psicología",
    "2": "Religión",
    "3": "Ciencias Sociales",
    "4": "Lenguas",
    "5": "Ciencias Puras",
    "6": "Ciencias Aplicadas",
    "7": "Artes y Recreación",
    "8": "Literatura",
    "9": "Historia y Geografía",
}

def reclasificar_linea(linea: str) -> str:
    partes = [p.strip() for p in linea.split("|")]
    if len(partes) < 3:
        return None  # línea inválida

    titulo, dewey, descripcion = partes[0], partes[1], partes[2]

    if not dewey or not dewey[0].isdigit():
        return f"{titulo} | Sin Dewey | Sin categoría | {descripcion}"

    categoria = dewey.strip()[0]  # primer dígito
    categoria_general = f"{categoria}00"
    nombre_categoria = DEWEY_GENERAL.get(categoria, "Desconocida")

    return f"{titulo} | {categoria_general} | {nombre_categoria} | {descripcion}"

def main(input_file="libros_normalizados.txt", output_file="libros_reclasificados.txt"):
    with open(input_file, "r", encoding="utf-8") as fin:
        lineas = fin.readlines()

    nuevas_lineas = []
    for linea in lineas:
        reclasificada = reclasificar_linea(linea.strip())
        if reclasificada:
            nuevas_lineas.append(reclasificada)

    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write("\n".join(nuevas_lineas))

    print(f"✅ Archivo generado: {output_file}")
    print(f"Total de líneas procesadas: {len(nuevas_lineas)}")

if __name__ == "__main__":
    main()
