#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrena un clasificador de categorías Dewey a partir de un TXT con líneas:
titulo | dewey | descripcion

- Filtra entradas con 'sin dewey' (case-insensitive)
- Preprocesa texto en español
- Permite elegir campos de texto: title, desc o ambos
- Divide en train/valid estratificado (con manejo seguro de clases raras)
- Entrena TF-IDF + LogisticRegression con class_weight='balanced'
- Guarda artefactos en ./artifacts_ml/

Uso:
    python train_dewey_model.py --data ./libros.txt --text_fields both --test_size 0.15

Requisitos:
    pip install scikit-learn joblib pandas numpy
"""

import os, re, json, argparse, unicodedata, warnings
from typing import Tuple
from collections import Counter
from math import ceil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from joblib import dump

# ---------------------------
# Utils
# ---------------------------

def strip_accents(text: str) -> str:
    """Quita acentos para robustez (opcional)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join([c for c in text if not unicodedata.combining(c)])

def basic_clean(text: str) -> str:
    """Limpieza liviana conservando señal semántica."""
    if not isinstance(text, str):
        return ""
    # elimina zero-width y normaliza espacios
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_line(line: str) -> Tuple[str, str, str]:
    """
    Espera formato: titulo | dewey | descripcion
    Devuelve (title, dewey, desc) ya limpiados.
    Soporta '|' dentro de la descripción.
    """
    # tolera BOM y comentarios
    line = line.lstrip("\ufeff")
    if line.strip().startswith("#"):
        return "", "", ""
    parts = [p.strip() for p in line.rstrip("\n").split("|", 2)]
    if len(parts) == 0:
        return "", "", ""
    if len(parts) == 1:
        parts += [""]
    if len(parts) == 2:
        parts += [""]
    title, dewey, desc = parts[0], parts[1], parts[2]
    return basic_clean(title), basic_clean(dewey), basic_clean(desc)

def is_missing_label(lbl: str) -> bool:
    """Detecta etiqueta faltante tipo 'sin dewey' (robusto a espacios/case)."""
    if not isinstance(lbl, str):
        return True
    s = basic_clean(lbl).casefold()
    return s == "" or s == "sin dewey"

# ---------------------------
# Carga y preparación
# ---------------------------

def load_dataset(txt_path: str) -> pd.DataFrame:
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"No existe el archivo: {txt_path}")

    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            t, d, s = parse_line(raw)
            if t == "" and d == "" and s == "":
                continue
            rows.append({"title": t, "dewey": d, "desc": s})

    if not rows:
        return pd.DataFrame(columns=["title", "dewey", "desc"])

    df = pd.DataFrame(rows)

    # Filtrar etiquetas faltantes
    mask_valid = ~df["dewey"].apply(is_missing_label)
    df = df[mask_valid].copy()

    # Quitar vacíos en etiqueta
    df = df[df["dewey"].astype(str).str.len() > 0]
    # Normalizar etiqueta
    df["dewey"] = df["dewey"].astype(str).str.strip()

    # Quitar filas sin texto (según campos elegidos más adelante se validará otra vez)
    df["title"] = df["title"].fillna("").astype(str)
    df["desc"] = df["desc"].fillna("").astype(str)

    return df.reset_index(drop=True)

def build_text_column(df: pd.DataFrame, text_fields: str) -> pd.Series:
    tf = text_fields.lower()
    if tf == "title":
        X = df["title"].fillna("")
    elif tf == "desc":
        X = df["desc"].fillna("")
    elif tf == "both":
        X = (df["title"].fillna("") + " [SEP] " + df["desc"].fillna(""))
    else:
        raise ValueError("text_fields debe ser: title | desc | both")
    # Por si quedaran vacíos absolutos
    return X.replace({None: ""}).astype(str)

# ---------------------------
# Modelo
# ---------------------------

def build_pipeline(ngram_max=3, min_df=2, max_df=0.9) -> Pipeline:
    """
    Pipeline: TF-IDF (palabras) + Logistic Regression.
    Mantengo simple y muy estable para minimizar fallas.
    """
    vectorizer = TfidfVectorizer(
        strip_accents=None,   # tenemos strip_accents() si algún día se quiere usar manual
        lowercase=True,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        n_jobs=None,  # ignorado por lbfgs, pero compatible con versiones modernas
        verbose=0,
    )
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    return pipe

# ---------------------------
# Entrenamiento
# ---------------------------

def train_and_eval(
    df: pd.DataFrame,
    text_fields: str,
    test_size: float,
    random_state: int,
    artifacts_dir: str,
    min_per_class: int = 2
):
    os.makedirs(artifacts_dir, exist_ok=True)

    # Semillas para reproducibilidad
    np.random.seed(random_state)

    X_all = build_text_column(df, text_fields)
    y_all = df["dewey"].astype(str).values
    idx_all = np.arange(len(df))

    # Si todo el texto está vacío, abortar con mensaje claro
    if (X_all.str.len() == 0).all():
        raise SystemExit("Todos los textos quedaron vacíos tras la limpieza. Revisa el archivo de entrada.")

    # separar clases raras (freq < min_per_class)
    cnt = Counter(y_all)
    mask_common = np.array([cnt[y] >= min_per_class for y in y_all])
    idx_common = idx_all[mask_common]
    idx_rare   = idx_all[~mask_common]

    # resolver tamaño de test si viene como fracción o entero
    def resolve_n_test(ts, n):
        if 0 < ts < 1:
            return max(1, min(n - 1, ceil(ts * n)))
        ts = int(ts)
        return max(1, min(n - 1, ts))

    # si no hay suficientes comunes para estratificar, split simple
    if len(idx_common) == 0 or len(set(y_all[idx_common])) <= 1:
        n_test = resolve_n_test(test_size, len(idx_all))
        perm = np.random.permutation(idx_all)
        idx_test = perm[:n_test]
        idx_train = perm[n_test:]
    else:
        # estratificar SOLO en comunes
        X_common = X_all.iloc[idx_common]
        y_common = y_all[idx_common]
        n_test_common = resolve_n_test(test_size, len(idx_common))
        # Para mantener estratificación estable usamos proporción en split
        test_prop_common = n_test_common / len(idx_common)
        (X_train_c, X_val_c,
         y_train_c, y_val_c,
         idx_train_c, idx_val_c) = train_test_split(
            X_common, y_common, idx_common,
            test_size=test_prop_common,
            random_state=random_state,
            stratify=y_common
        )
        # raras -> SOLO train
        idx_train = np.concatenate([idx_train_c, idx_rare], axis=0)
        idx_test  = idx_val_c

    # construir splits finales
    X_train, y_train = X_all.iloc[idx_train], y_all[idx_train]
    X_val,   y_val   = X_all.iloc[idx_test],  y_all[idx_test]

    # Label maps (sobre train)
    labels = sorted(pd.Series(y_train).unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    # Entrenar
    warnings.filterwarnings("ignore")  # silenciar ConvergenceWarnings si aparecen
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_hat = pipe.predict(X_val)

    # Métricas
    acc = accuracy_score(y_val, y_hat)
    f1_macro = f1_score(y_val, y_hat, average="macro")
    bal_acc = balanced_accuracy_score(y_val, y_hat)
    report = classification_report(y_val, y_hat, output_dict=True, zero_division=0)
    # matriz de confusión SOLO de clases vistas en train
    cm = confusion_matrix(y_val, y_hat, labels=labels)

    # Guardar artefactos
    dump(pipe, os.path.join(artifacts_dir, "model.joblib"))
    with open(os.path.join(artifacts_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(artifacts_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    metrics = {
        "num_samples": int(len(df)),
        "num_train": int(len(X_train)),
        "num_val": int(len(X_val)),
        "num_classes_train": int(len(labels)),
        "classes_train": labels,
        "text_fields": text_fields,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "report": report,
        "rare_in_train_only": int(len(idx_rare)),
        "random_state": random_state,
        "test_size_input": test_size,
        "min_per_class": min_per_class
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(artifacts_dir, "confusion_matrix.csv"), encoding="utf-8")

    # también guardo métricas por-clase en CSV para inspección rápida
    per_class_rows = []
    for lbl in labels:
        r = report.get(lbl, {})
        per_class_rows.append({
            "label": lbl,
            "precision": r.get("precision", 0.0),
            "recall": r.get("recall", 0.0),
            "f1": r.get("f1-score", 0.0),
            "support": int(r.get("support", 0)),
        })
    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(artifacts_dir, "per_class_report.csv"),
        index=False, encoding="utf-8"
    )

    print("✅ Entrenamiento finalizado.")
    print(f"- Total: {len(df)} | Train: {len(X_train)} | Val: {len(X_val)} | Clases en train: {len(labels)}")
    print(f"- Rare-only-to-train: {int(len(idx_rare))}")
    print(f"- Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f} | Balanced-Acc: {bal_acc:.4f}")
    if len(labels) <= 20:
        print(f"- Clases: {labels}")

# ---------------------------
# Main / CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Ruta al TXT: titulo|dewey|descripcion")
    parser.add_argument("--text_fields", type=str, default="both", choices=["title","desc","both"],
                        help="Campos a usar como texto de entrada")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción (0-1) o entero para validación")
    parser.add_argument("--random_state", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_ml", help="Carpeta de salida")
    parser.add_argument("--min_per_class", type=int, default=2, help="Mínimo por clase para permitir estratificación")

    args = parser.parse_args()

    print("Cargando y limpiando datos...")
    df = load_dataset(args.data)
    if df.empty:
        raise SystemExit("No hay datos válidos luego de filtrar 'sin dewey' o el archivo está vacío.")

    print(f"Total de muestras válidas: {len(df)} | Clases únicas: {df['dewey'].nunique()}")

    train_and_eval(
        df=df,
        text_fields=args.text_fields,
        test_size=args.test_size,
        random_state=args.random_state,
        artifacts_dir=args.artifacts_dir,
        min_per_class=args.min_per_class
    )

if __name__ == "__main__":
    main()
