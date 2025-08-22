#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrena un clasificador de categorías Dewey a partir de un TXT con líneas:
titulo | dewey | descripcion

Características:
- Representación híbrida: TF-IDF de palabras (1–3) + caracteres (3–5)
- Stopwords multilingüe: none | es | en | both | file
- GridSearchCV (macro-F1) con StratifiedKFold (opcional con --tune)
- Calibración de probabilidades (opcional con --calibrate)
- Persistencia de resultados y metadatos

Uso:
    python train_dewey_model.py --data ./libros.txt --text_fields both --test_size 0.15 --tune --calibrate

Requisitos:
    pip install scikit-learn joblib pandas numpy
"""

import os, re, json, argparse, warnings
from typing import Tuple, Optional, List
from collections import Counter
from math import ceil
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

# ---------------------------
# Utils
# ---------------------------

def basic_clean(text: str) -> str:
    """Limpieza liviana conservando señal semántica."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_line(line: str) -> Tuple[str, str, str]:
    """
    Espera formato: titulo | dewey | descripcion
    Devuelve (title, dewey, desc) ya limpiados.
    Soporta '|' dentro de la descripción.
    """
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
# Stopwords multilingüe
# ---------------------------

def stopwords_es_basic() -> List[str]:
    """Lista mínima de stopwords en español (editable)."""
    return [
        "a","acá","ahora","al","algo","algún","alguna","algunas","alguno","algunos","allí","ante",
        "antes","aquel","aquella","aquellas","aquello","aquellos","aquí","así","aún","cada","como",
        "con","contra","cual","cuales","cualquier","cualquiera","de","del","desde","donde","dos",
        "el","él","ella","ellas","ello","ellos","en","entre","era","erais","éramos","eran","eres",
        "es","esa","esas","ese","eso","esos","esta","estaba","estabais","estábamos","estaban","estado",
        "estáis","estamos","están","estar","este","esto","estos","ex","fue","fueron","fui","fuimos",
        "ha","habeis","habéis","había","habían","han","hasta","hay","la","las","le","les","lo","los",
        "más","me","mi","mis","mucha","muchas","mucho","muchos","muy","nada","ni","no","nos","nosotros",
        "nuestra","nuestras","nuestro","nuestros","o","os","otra","otras","otro","otros","para","pero",
        "poco","por","porque","qué","que","se","sea","ser","si","sí","sin","sobre","su","sus","tal",
        "también","tanto","te","tenéis","tenemos","tiene","tienen","todo","todos","tras","tu","tus",
        "un","una","uno","unos","vosotros","vuestra","vuestras","vuestro","vuestros","ya","y"
    ]

def resolve_stopwords(mode: str, path: Optional[str]=None) -> Optional[List[str]]:
    """
    Devuelve stopwords según modo:
      - 'none' -> None
      - 'es'   -> lista embebida ES
      - 'en'   -> ENGLISH_STOP_WORDS de scikit-learn
      - 'both' -> unión ES ∪ EN
      - 'file' -> carga desde archivo (una palabra por línea)
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    if mode == "es":
        return stopwords_es_basic()
    if mode == "en":
        return list(ENGLISH_STOP_WORDS)
    if mode == "both":
        return sorted(set(stopwords_es_basic()).union(set(ENGLISH_STOP_WORDS)))
    if mode == "file":
        if not path:
            raise ValueError("Debe proveer --stopwords_file cuando --stopwords file")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe el archivo de stopwords: {path}")
        words = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    words.append(w)
        return words or None
    raise ValueError("stopwords debe ser: none | es | en | both | file")

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
    mask_valid = ~df["dewey"].apply(is_missing_label)
    df = df[mask_valid].copy()

    df = df[df["dewey"].astype(str).str.len() > 0]
    df["dewey"] = df["dewey"].astype(str).str.strip()
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
    return X.replace({None: ""}).astype(str)

# ---------------------------
# Modelos / Pipelines
# ---------------------------

def build_feature_union(
    ngram_word_max=3, ngram_char_min=3, ngram_char_max=5,
    min_df=2, max_df=0.9, stop_words=None, strip_accents: Optional[str]=None
) -> FeatureUnion:
    """
    Representación híbrida: palabras + caracteres.
    - Palabras: capta semántica; admite stopwords multilingüe y strip_accents.
    - Caracteres: robustece ante errores tipográficos y variantes morfológicas.
    """
    tfidf_word = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, ngram_word_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        stop_words=stop_words,           # lista o None
        strip_accents=strip_accents,     # 'unicode' o None (según CLI)
    )
    tfidf_char = TfidfVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(ngram_char_min, ngram_char_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents=strip_accents,
    )
    return FeatureUnion([("w", tfidf_word), ("c", tfidf_char)])

def build_base_pipeline(stop_words=None, strip_accents: Optional[str]=None) -> Pipeline:
    feats = build_feature_union(stop_words=stop_words, strip_accents=strip_accents)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        verbose=0,
    )
    pipe = Pipeline([("feats", feats), ("clf", clf)])
    return pipe

def get_param_grid() -> dict:
    """
    Espacio de búsqueda razonable y compacto para LR y TF-IDF.
    """
    return {
        "feats__w__ngram_range": [ (1,2), (1,3) ],
        "feats__w__min_df": [1, 2, 3],
        "feats__w__max_df": [0.85, 0.9, 0.95],
        "feats__c__ngram_range": [ (3,5), (4,6) ],
        "feats__c__min_df": [1, 2, 3],
        "clf__C": [0.5, 1.0, 2.0, 4.0],
    }

# ---------------------------
# Entrenamiento
# ---------------------------

def train_and_eval(
    df: pd.DataFrame,
    text_fields: str,
    test_size: float,
    random_state: int,
    artifacts_dir: str,
    min_per_class: int = 2,
    tune: bool = False,
    calibrate: bool = False,
    cv_folds: int = 5,
    stopwords_mode: str = "both",
    stopwords_file: Optional[str] = None,
    strip_accents: Optional[str] = None
):
    os.makedirs(artifacts_dir, exist_ok=True)
    np.random.seed(random_state)

    X_all = build_text_column(df, text_fields)
    y_all = df["dewey"].astype(str).values
    idx_all = np.arange(len(df))

    if (X_all.str.len() == 0).all():
        raise SystemExit("Todos los textos quedaron vacíos tras la limpieza. Revisa el archivo de entrada.")

    # Clases raras (freq < min_per_class): sólo a train
    cnt = Counter(y_all)
    mask_common = np.array([cnt[y] >= min_per_class for y in y_all])
    idx_common = idx_all[mask_common]
    idx_rare   = idx_all[~mask_common]

    def resolve_n_test(ts, n):
        if 0 < ts < 1:
            return max(1, min(n - 1, ceil(ts * n)))
        ts = int(ts)
        return max(1, min(n - 1, ts))

    if len(idx_common) == 0 or len(set(y_all[idx_common])) <= 1:
        n_test = resolve_n_test(test_size, len(idx_all))
        perm = np.random.permutation(idx_all)
        idx_test = perm[:n_test]
        idx_train = perm[n_test:]
    else:
        X_common = X_all.iloc[idx_common]
        y_common = y_all[idx_common]
        n_test_common = resolve_n_test(test_size, len(idx_common))
        test_prop_common = n_test_common / len(idx_common)
        (X_train_c, X_val_c,
         y_train_c, y_val_c,
         idx_train_c, idx_val_c) = train_test_split(
            X_common, y_common, idx_common,
            test_size=test_prop_common,
            random_state=random_state,
            stratify=y_common
        )
        idx_train = np.concatenate([idx_train_c, idx_rare], axis=0)
        idx_test  = idx_val_c

    X_train, y_train = X_all.iloc[idx_train], y_all[idx_train]
    X_val,   y_val   = X_all.iloc[idx_test],  y_all[idx_test]

    labels = sorted(pd.Series(y_train).unique().tolist())

    # === Entrenamiento ===
    warnings.filterwarnings("ignore")

    sw = resolve_stopwords(stopwords_mode, stopwords_file)
    if strip_accents is not None:
        strip_accents = strip_accents.lower()
        if strip_accents not in {"none", "unicode"}:
            raise ValueError("--strip_accents debe ser none o unicode")
        strip_accents = None if strip_accents == "none" else "unicode"

    pipe = build_base_pipeline(stop_words=sw, strip_accents=strip_accents)

    if tune:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=get_param_grid(),
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        # Persistencia CV
        pd.DataFrame(grid.cv_results_).to_csv(
            os.path.join(artifacts_dir, "cv_results.csv"), index=False, encoding="utf-8"
        )
        with open(os.path.join(artifacts_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(grid.best_params_, f, ensure_ascii=False, indent=2)
        model_to_use = best_pipe
    else:
        model_to_use = pipe.fit(X_train, y_train)

    # Calibración de probabilidades (opcional)
    if calibrate:
        calibrator = CalibratedClassifierCV(base_estimator=model_to_use, method="sigmoid", cv=3)
        calibrator.fit(X_train, y_train)
        final_model = calibrator
    else:
        final_model = model_to_use

    # Evaluación en VAL
    y_hat = final_model.predict(X_val)

    acc = accuracy_score(y_val, y_hat)
    f1_macro = f1_score(y_val, y_hat, average="macro")
    bal_acc = balanced_accuracy_score(y_val, y_hat)
    report = classification_report(y_val, y_hat, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_hat, labels=labels)

    # Guardado de artefactos
    dump(final_model, os.path.join(artifacts_dir, "model.joblib"))

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    with open(os.path.join(artifacts_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(artifacts_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
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
        "random_state": random_state,
        "test_size_input": test_size,
        "min_per_class": min_per_class,
        "tuned": bool(tune),
        "calibrated": bool(calibrate),
        "cv_folds": int(cv_folds) if tune else None,
        "stopwords_mode": stopwords_mode,
        "strip_accents": ("unicode" if strip_accents else "none")
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Matriz de confusión y per-class report
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(artifacts_dir, "confusion_matrix.csv"), encoding="utf-8")

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

    # Metadatos de versión/config
    version = {
        "script": "train_dewey_model.py",
        "scikit_learn": __import__("sklearn").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pipeline": "FeatureUnion(TFIDF word + char) + LogisticRegression",
        "notes": "Stopwords ES/EN/Both; strip_accents opcional; GridSearch y calibración opcionales."
    }
    with open(os.path.join(artifacts_dir, "VERSION.json"), "w", encoding="utf-8") as f:
        json.dump(version, f, ensure_ascii=False, indent=2)

    print("✅ Entrenamiento finalizado.")
    print(f"- Total: {len(df)} | Train: {len(X_train)} | Val: {len(X_val)} | Clases en train: {len(labels)}")
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

    # Flags de entrenamiento
    parser.add_argument("--tune", action="store_true", help="Activa GridSearchCV (recomendado para datasets pequeños/medianos)")
    parser.add_argument("--calibrate", action="store_true", help="Calibra probabilidades con sigmoid")
    parser.add_argument("--cv_folds", type=int, default=5, help="Folds para CV si --tune")

    # Multilingüe
    parser.add_argument("--stopwords", type=str, default="both",
                        choices=["none","es","en","both","file"],
                        help="Estrategia de stopwords")
    parser.add_argument("--stopwords_file", type=str, default=None,
                        help="Ruta a archivo de stopwords si --stopwords file")
    parser.add_argument("--strip_accents", type=str, default="unicode",
                        choices=["none","unicode"],
                        help="Normalización de acentos en TF-IDF")

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
        min_per_class=args.min_per_class,
        tune=args.tune,
        calibrate=args.calibrate,
        cv_folds=args.cv_folds,
        stopwords_mode=args.stopwords,
        stopwords_file=args.stopwords_file,
        strip_accents=args.strip_accents
    )

if __name__ == "__main__":
    main()
