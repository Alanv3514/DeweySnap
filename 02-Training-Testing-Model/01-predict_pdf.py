#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clasifica un PDF usando el pipeline entrenado (model.joblib).

- Extrae texto del PDF (pypdf)
- Preprocesa con la misma limpieza básica del training
- Divide en chunks y predice por chunk
- Agrega por probas promedio para decisión final
- (Opcional) Generaliza la clase Dewey a 000–900
- Guarda un reporte JSON

Uso:
    python predict_pdf.py --pdf ./ejemplo.pdf --artifacts ./artifacts_ml --text_fields both --top_k 5 --generalize
"""

import os, re, json, argparse, unicodedata
from typing import List, Dict, Any
from joblib import load
from pypdf import PdfReader
import numpy as np

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

# --------- limpieza igual que en training ---------
def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join([c for c in text if not unicodedata.combining(c)])

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------- extracción de texto PDF ---------
def extract_pdf_text(path: str, max_pages: int = None) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el PDF: {path}")
    reader = PdfReader(path)
    pages = min(len(reader.pages), max_pages) if max_pages else len(reader.pages)
    chunks = []
    for i in range(pages):
        try:
            txt = reader.pages[i].extract_text() or ""
        except Exception:
            txt = ""
        chunks.append(txt)
    return "\n".join(chunks)

# --------- preparación de texto según text_fields ---------
def build_text_for_inference(text: str, text_fields: str = "both") -> str:
    """
    En training, 'both' unía title + desc con [SEP].
    Para PDF no hay título/desc separados: simulamos 'both' como '[DOC] [SEP] <texto>'
    para no romper el vectorizador entrenado.
    """
    t = basic_clean(text)
    if text_fields == "both":
        return f"[DOC] [SEP] {t}"
    return t

# --------- chunking ---------
def chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        add = len(w) + 1
        if cur_len + add > max_chars:
            chunks.append(" ".join(cur).strip())
            cur = [w]
            cur_len = add
        else:
            cur.append(w)
            cur_len += add
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

# --------- generalización Dewey ---------
def generalize_dewey(label: str) -> Dict[str, str]:
    if not label or not label[0].isdigit():
        return {"code": "Sin Dewey", "name": "Sin categoría"}
    d0 = label.strip()[0]
    return {"code": f"{d0}00", "name": DEWEY_GENERAL.get(d0, "Desconocida")}

# --------- predicción ---------
def predict_pdf(pdf_path: str, artifacts_dir: str, text_fields: str = "both",
                max_pages: int = None, top_k: int = 5, max_chars_chunk: int = 8000,
                generalize: bool = False) -> Dict[str, Any]:
    model_path = os.path.join(artifacts_dir, "model.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    pipe = load(model_path)

    # extraer texto
    raw_text = extract_pdf_text(pdf_path, max_pages=max_pages)
    text_for_model = build_text_for_inference(raw_text, text_fields=text_fields)
    chunks = chunk_text(text_for_model, max_chars=max_chars_chunk)
    if not chunks:
        raise SystemExit("El PDF no contiene texto extraíble tras la limpieza.")

    supports_proba = hasattr(pipe, "predict_proba")
    classes = getattr(pipe, "classes_", None)

    if not supports_proba:
        preds = pipe.predict(chunks)
        labels = np.array(preds)
        winner_label = labels[0]
        out = {
            "chunks": [{"idx": i, "pred": str(preds[i])} for i in range(len(preds))],
            "final_pred": str(winner_label),
            "final_score": None,
            "top_k": None,
        }
        if generalize:
            out["final_general"] = generalize_dewey(out["final_pred"])
        return out

    proba = pipe.predict_proba(chunks)  # (n_chunks, n_classes)
    if classes is None:
        raise RuntimeError("El pipeline no expone 'classes_' para mapear índices a etiquetas.")

    avg_proba = proba.mean(axis=0)
    order = np.argsort(avg_proba)[::-1]

    winner_idx = int(order[0])
    winner_label = str(classes[winner_idx])
    winner_score = float(avg_proba[winner_idx])

    k = min(top_k, len(order))
    top_list = [{"label": str(classes[int(i)]), "score": float(avg_proba[int(i)])} for i in order[:k]]

    per_chunk = []
    for i in range(len(chunks)):
        pc_idx = int(np.argmax(proba[i]))
        per_chunk.append({
            "idx": i,
            "pred": str(classes[pc_idx]),
            "score": float(proba[i][pc_idx]),
            "chars": len(chunks[i])
        })

    out = {
        "final_pred": winner_label,
        "final_score": winner_score,
        "top_k": top_list,
        "chunks": per_chunk
    }
    if generalize:
        out["final_general"] = generalize_dewey(out["final_pred"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Ruta al PDF a clasificar")
    ap.add_argument("--artifacts", default="./artifacts_ml", help="Carpeta con model.joblib y mapas")
    ap.add_argument("--text_fields", default="both", choices=["title", "desc", "both"],
                    help="Debe coincidir con lo usado en entrenamiento")
    ap.add_argument("--max_pages", type=int, default=None, help="Limitar páginas (opcional)")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K clases a reportar")
    ap.add_argument("--max_chars_chunk", type=int, default=8000, help="Tamaño de chunk en chars")
    ap.add_argument("--report_json", default="prediction_report.json", help="Ruta de salida del reporte JSON")
    ap.add_argument("--generalize", action="store_true", help="Devolver también la categoría 000–900")
    args = ap.parse_args()

    res = predict_pdf(
        pdf_path=args.pdf,
        artifacts_dir=args.artifacts,
        text_fields=args.text_fields,
        max_pages=args.max_pages,
        top_k=args.top_k,
        max_chars_chunk=args.max_chars_chunk,
        generalize=args.generalize
    )

    # Guardar reporte JSON
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
