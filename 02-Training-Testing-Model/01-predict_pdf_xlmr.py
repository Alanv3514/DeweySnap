#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clasifica un PDF usando el modelo Hugging Face fine-tuneado (artifacts_hf).

- Extrae texto del PDF (pypdf)
- Preprocesa con la misma limpieza básica del training
- Divide en chunks y predice por chunk con el modelo HF
- Agrega por probabilidades promedio para la decisión final
- (Opcional) Generaliza la clase Dewey a 000–900
- Guarda un reporte JSON

Uso:
    python predict_pdf_hf.py --pdf ./ejemplo.pdf --artifacts ./artifacts_hf --text_fields both --top_k 5 --generalize
"""

import os, re, json, argparse, unicodedata
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

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
    para no romper el patrón del entrenamiento.
    """
    t = basic_clean(text)
    if text_fields == "both":
        return f"[DOC] [SEP] {t}"
    return t  # 'title' o 'desc' se comportan igual aquí

# --------- chunking sencillo por caracteres ---------
def chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    chunks, cur, cur_len = [], [], 0
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

# --------- utilidades de labels (HF config o JSON) ---------
def load_labels(artifacts_dir: str, model) -> List[str]:
    # 1) Intentar desde config.id2label (dict {int: str} o {str(int): str})
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        # ordenar por id numérico
        pairs: List[Tuple[int, str]] = []
        for k, v in id2label.items():
            try:
                ki = int(k)
            except Exception:
                # algunas configs ya tienen clave int
                ki = int(k) if not isinstance(k, int) else k
            pairs.append((ki, str(v)))
        pairs.sort(key=lambda x: x[0])
        return [lab for _, lab in pairs]

    # 2) Fallback: id2label.json en el directorio
    json_path = os.path.join(artifacts_dir, "id2label.json")
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # viene como { "0": "001", "1": "100", ... } → ordenar por clave
        return [d[str(i)] for i in sorted(map(int, d.keys()))]

    # 3) Último recurso: deducir num_labels y usar índices como strings
    num_labels = getattr(model.config, "num_labels", None)
    if isinstance(num_labels, int) and num_labels > 0:
        return [str(i) for i in range(num_labels)]
    raise RuntimeError("No se pudieron determinar las etiquetas (id2label).")

# --------- predicción HF ---------
def predict_pdf(pdf_path: str,
                artifacts_dir: str,
                text_fields: str = "both",
                max_pages: int = None,
                top_k: int = 5,
                max_chars_chunk: int = 8000,
                generalize: bool = False,
                max_length: int = 384,
                batch_size_infer: int = 8,
                use_amp: bool = True) -> Dict[str, Any]:

    if not os.path.isdir(artifacts_dir):
        raise FileNotFoundError(f"No existe el directorio de artefactos HF: {artifacts_dir}")

    # Cargar modelo/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifacts_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # etiquetas (ordenadas por id)
    labels = load_labels(artifacts_dir, model)
    num_labels = len(labels)

    # extraer texto
    raw_text = extract_pdf_text(pdf_path, max_pages=max_pages)
    text_for_model = build_text_for_inference(raw_text, text_fields=text_fields)
    chunks = chunk_text(text_for_model, max_chars=max_chars_chunk)
    if not chunks:
        raise SystemExit("El PDF no contiene texto extraíble tras la limpieza.")

    # inferencia por lotes
    all_probs = []
    per_chunk = []

    autocast_dtype = torch.float16 if (device.type == "cuda" and use_amp) else None
    for i in range(0, len(chunks), batch_size_infer):
        batch_texts = chunks[i:i + batch_size_infer]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    outputs = model(**enc)
            else:
                outputs = model(**enc)

        logits = outputs.logits  # (B, num_labels)
        probs = F.softmax(logits.float(), dim=-1)  # a float32 en CPU
        probs_np = probs.detach().cpu().numpy()
        all_probs.append(probs_np)

        # por-chunk
        tops = probs_np.argmax(axis=1)
        scores = probs_np.max(axis=1)
        for j, (tidx, sc) in enumerate(zip(tops, scores)):
            per_chunk.append({
                "idx": i + j,
                "pred": str(labels[int(tidx)]),
                "score": float(sc),
                "chars": len(batch_texts[j])
            })

    all_probs = np.vstack(all_probs)  # (n_chunks, num_labels)

    # agregado: promedio simple de probabilidades por clase
    avg_proba = all_probs.mean(axis=0)  # (num_labels,)
    order = np.argsort(avg_proba)[::-1]
    k = min(top_k, num_labels)

    final_idx = int(order[0])
    final_label = str(labels[final_idx])
    final_score = float(avg_proba[final_idx])

    top_list = [
        {"label": str(labels[int(idx)]), "score": float(avg_proba[int(idx)])}
        for idx in order[:k]
    ]

    out = {
        "final_pred": final_label,
        "final_score": final_score,
        "top_k": top_list,
        "chunks": per_chunk,
        "num_chunks": len(chunks),
        "num_labels": num_labels
    }
    if generalize:
        out["final_general"] = generalize_dewey(final_label)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Ruta al PDF a clasificar")
    ap.add_argument("--artifacts", default="./artifacts_hf", help="Carpeta del modelo/tokenizer HF")
    ap.add_argument("--text_fields", default="both", choices=["title", "desc", "both"],
                    help="Debe coincidir con lo usado en entrenamiento (sug. both)")
    ap.add_argument("--max_pages", type=int, default=None, help="Limitar páginas (opcional)")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K clases a reportar")
    ap.add_argument("--max_chars_chunk", type=int, default=8000, help="Tamaño de chunk en chars")
    ap.add_argument("--report_json", default="prediction_report.json", help="Ruta de salida del reporte JSON")
    ap.add_argument("--generalize", action="store_true", help="Devolver también la categoría 000–900")
    ap.add_argument("--max_length", type=int, default=384, help="max_length del tokenizer")
    ap.add_argument("--batch_size_infer", type=int, default=8, help="Batch de inferencia (chunks por lote)")
    ap.add_argument("--no_amp", action="store_true", help="Desactivar autocast FP16 en CUDA")
    args = ap.parse_args()

    res = predict_pdf(
        pdf_path=args.pdf,
        artifacts_dir=args.artifacts,
        text_fields=args.text_fields,
        max_pages=args.max_pages,
        top_k=args.top_k,
        max_chars_chunk=args.max_chars_chunk,
        generalize=args.generalize,
        max_length=args.max_length,
        batch_size_infer=args.batch_size_infer,
        use_amp=(not args.no_amp),
    )

    # Guardar reporte JSON
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
