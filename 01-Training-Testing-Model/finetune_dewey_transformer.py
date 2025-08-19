#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning de un Transformer en español para clasificar títulos/libros en categorías Dewey (nivel centena).
Entrada esperada por línea en TXT: nombre | dewey | (anio o resumen)
- Si la 3ra columna NO parece un año (19xx/20xx), se concatena al título como texto adicional.

Salidas:
- ./artifacts_hf/
    - config.json, pytorch_model.bin, tokenizer.*, special_tokens_map.json
    - label2id.json, id2label.json
    - metrics.json

Uso mínimo:
    python finetune_dewey_transformer.py --data ./libros.txt

Recomendado (GPU si disponible):
    accelerate launch finetune_dewey_transformer.py --data ./libros.txt --model PlanTL-GOB-ES/roberta-base-bne --epochs 4 --batch_size 16
"""

import os, re, json, random, argparse, unicodedata
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter

import torch
from torch import nn

from datasets import Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)

# ------------------ Utilidades ------------------

YEAR_RE = re.compile(r"^\s*(19|20)\d{2}\s*$")

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def normalize_dewey_centena(dewey: str) -> str:
    """Devuelve '000','100',... según el primer dígito encontrado. Fallback '000'."""
    if not dewey:
        return "000"
    m = re.search(r"(\d)", dewey)
    if not m:
        return "000"
    d = int(m.group(1))
    return f"{d}00"

def parse_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip().strip('"').strip("'") for p in line.split("|", maxsplit=2)]
    if len(parts) < 2:
        return None
    nombre = parts[0]
    dewey = parts[1]
    extra = parts[2] if len(parts) > 2 else ""
    return (nombre, dewey, extra)

def load_txt(path: str) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (texto, etiqueta_centena).
    texto = nombre si 3ra col es año; de lo contrario "nombre. extra".
    """
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parsed = parse_line(raw)
            if not parsed:
                continue
            nombre, dewey, extra = parsed
            etiqueta = normalize_dewey_centena(dewey)
            if YEAR_RE.match(extra or ""):
                texto = nombre
            else:
                texto = f"{nombre}. {extra}" if extra else nombre
            data.append((texto, etiqueta))
    return data

def build_label_maps(y: List[str]):
    classes = sorted(list(set(y)))
    label2id = {c:i for i,c in enumerate(classes)}
    id2label = {i:c for c,i in label2id.items()}
    return label2id, id2label, classes

# ------------------ Trainer con pesos ------------------

class WeightedTrainer(Trainer):
    """Incorpora weights de clase en la función de pérdida si se proporcionan."""
    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ------------------ Métricas ------------------

def build_compute_metrics(id2label):
    metric_acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": f1_macro}
    return compute_metrics

# ------------------ Tokenización ------------------

@dataclass
class PreprocessFn:
    tokenizer: AutoTokenizer
    def __call__(self, examples):
        # Limpieza ligera; no quitamos acentos (el tokenizer BPE/WordPiece los maneja)
        texts = examples["text"]
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=128
        )

# ------------------ Inferencia PDF ------------------

def titulo_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        r = PdfReader(path)
        t = (r.metadata.title or "").strip() if r.metadata else ""
        if not t:
            text = (r.pages[0].extract_text() or "").strip()
            t = " ".join(text.split()[:16])
        return t
    except Exception:
        return ""

def predict_title(model_dir: str, title: str, topk=3):
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    encoded = tok(title, return_tensors="pt", truncation=True, max_length=128)
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    idxs = probs.argsort()[::-1][:topk]
    return [(id2label[i], float(probs[i])) for i in idxs]

# ------------------ Carga robusta de tokenizer/modelo ------------------

def load_tokenizer_and_model(model_id: str, classes: List[str], local_files_only: bool = False):
    """
    Intenta:
      1) use_fast=True
      2) use_fast=False
      3) Fallback a BETO (dccuchile/bert-base-spanish-wwm-cased)
    Configura id2label/label2id con las CLASES reales.
    """
    tried = []

    def _attempt(mid: str, use_fast: bool):
        tried.append(f"{mid} (use_fast={use_fast})")
        tok = AutoTokenizer.from_pretrained(mid, use_fast=use_fast, local_files_only=local_files_only)
        cfg = AutoConfig.from_pretrained(
            mid,
            num_labels=len(classes),
            id2label={i: l for i, l in enumerate(classes)},
            label2id={l: i for i, l in enumerate(classes)},
            local_files_only=local_files_only
        )
        mdl = AutoModelForSequenceClassification.from_pretrained(mid, config=cfg, local_files_only=local_files_only)
        return tok, cfg, mdl

    # 1) intentamos tokenizer rápido
    try:
        return _attempt(model_id, use_fast=True)
    except Exception as e_primary:
        print(f"[WARN] No se pudo cargar '{model_id}' con tokenizer rápido. Motivo: {e_primary}")

    # 2) intentamos tokenizer lento
    try:
        tok, cfg, mdl = _attempt(model_id, use_fast=False)
        print("[INFO] Cargado con tokenizer lento (use_fast=False).")
        return tok, cfg, mdl
    except Exception as e_slow:
        print(f"[WARN] Tampoco se pudo con tokenizer lento. Motivo: {e_slow}")

    # 3) Fallback a BETO
    beto = "dccuchile/bert-base-spanish-wwm-cased"
    try:
        print(f"[INFO] Probando fallback a '{beto}' ...")
        return _attempt(beto, use_fast=True)
    except Exception as e_beto:
        tried.append(f"{beto} (use_fast=True)")
        # Intento final con tokenizer lento en BETO
        try:
            return _attempt(beto, use_fast=False)
        except Exception as e_beto_slow:
            tried.append(f"{beto} (use_fast=False)")
            raise RuntimeError(
                "No se pudo cargar ningún tokenizer/modelo.\n"
                f"Intentos: {tried}\n"
                f"Último error: {e_beto_slow}\n"
                "Sugerencias:\n"
                "- Verifica la conexión a Internet, o usa --model apuntando a una carpeta local con los archivos del modelo.\n"
                "- Actualiza 'transformers' y 'tokenizers' a versiones recientes.\n"
                "- Revisa que el ID esté bien escrito."
            )

# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Ruta al TXT: nombre|dewey|extra")
    parser.add_argument("--model", type=str, default="PlanTL-GOB-ES/roberta-base-bne",
                        help="Modelo base (BETO: 'dccuchile/bert-base-spanish-wwm-cased')")
    parser.add_argument("--outdir", type=str, default="./artifacts_hf")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--early_patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--local_files_only", action="store_true",
                        help="Si se establece, carga solo desde archivos locales/cache (sin Internet).")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Cargar datos
    data = load_txt(args.data)
    if len(data) < 50:
        print("Advertencia: dataset pequeño; los resultados pueden ser inestables.")
    texts = [d[0] for d in data]
    labels_str = [d[1] for d in data]

    # 2) Mapas de etiquetas
    label2id, id2label, classes = build_label_maps(labels_str)
    y = np.array([label2id[s] for s in labels_str], dtype=np.int64)

    # 3) Split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=args.val_ratio, random_state=args.seed, stratify=y
    )

    # 4) Tokenizer y modelo con carga robusta
    tokenizer, config, model = load_tokenizer_and_model(
        args.model, classes=classes, local_files_only=args.local_files_only
    )

    # 5) Datasets tokenizados
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds   = Dataset.from_dict({"text": X_val,   "label": y_val})

    preprocess = PreprocessFn(tokenizer)
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["text"])
    val_tok   = val_ds.map(preprocess,   batched=True, remove_columns=["text"])

    # 6) Pesos de clase (inverso a la frecuencia)
    counts = Counter(y_train.tolist())
    class_weights = torch.tensor([1.0 / counts[i] for i in range(len(classes))], dtype=torch.float)

    # 7) Data collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8) Métricas
    compute_metrics = build_compute_metrics(config.id2label)

    # 9) Args de entrenamiento
    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=args.fp16,
        gradient_accumulation_steps=args.grad_accum,
        report_to="none"
    )

    # 10) Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_patience)]
    )

    # 11) Entrenar
    train_result = trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    # 12) Guardar
    trainer.save_model(args.outdir)  # guarda modelo + tokenizer
    tokenizer.save_pretrained(args.outdir)
    with open(os.path.join(args.outdir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in config.label2id.items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in config.id2label.items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 13) Ejemplo de inferencia directa
    ejemplo = "Redes neuronales aplicadas a la visión por computador"
    topk = predict_title(args.outdir, ejemplo, topk=3)
    print("Ejemplo inferencia:", ejemplo, "=>", topk)

if __name__ == "__main__":
    main()
