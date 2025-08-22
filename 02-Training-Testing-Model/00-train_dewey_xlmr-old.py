#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning multilingüe (ES/EN) para clasificar categorías Dewey desde un TXT:
titulo | dewey | descripcion

- Modelo: xlm-roberta-base (multilingüe)
- Manejo de clases raras: las manda a TRAIN (no a VALID)
- Split estratificado en clases "comunes"
- Tokenización sobre título/desc (uno, otro o ambos)
- Class weights para desbalanceo
- FP16 en GPU, EarlyStopping, mejor checkpoint
- Artefactos: modelo HF, label2id/id2label, metrics.json, reportes

Uso mínimo:
    python train_dewey_xlmr.py --data ./libros.txt --text_fields both

Recomendado (GPU):
    accelerate launch train_dewey_xlmr.py --data ./libros.txt --text_fields both --epochs 5 --batch_size 16
"""

import os, json, re, argparse, warnings
from typing import Tuple, List, Dict
from collections import Counter
from math import ceil
from datetime import datetime

import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback
)
from transformers.trainer import TrainerCallback
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
import inspect
# ---------------------------
# Utilidades de parsing y limpieza (idénticas a tu flujo)
# ---------------------------

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_line(line: str) -> Tuple[str, str, str]:
    line = line.lstrip("\ufeff")
    if line.strip().startswith("#"):
        return "", "", ""
    parts = [p.strip() for p in line.rstrip("\n").split("|", 2)]
    if len(parts) == 0: return "", "", ""
    if len(parts) == 1: parts += [""]
    if len(parts) == 2: parts += [""]
    title, dewey, desc = parts[0], parts[1], parts[2]
    return basic_clean(title), basic_clean(dewey), basic_clean(desc)

def is_missing_label(lbl: str) -> bool:
    if not isinstance(lbl, str): return True
    s = basic_clean(lbl).casefold()
    return s == "" or s == "sin dewey"

def load_dataset(txt_path: str) -> pd.DataFrame:
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"No existe el archivo: {txt_path}")
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip(): continue
            t, d, s = parse_line(raw)
            if t == "" and d == "" and s == "": continue
            rows.append({"title": t, "dewey": d, "desc": s})
    if not rows:
        return pd.DataFrame(columns=["title","dewey","desc"])
    df = pd.DataFrame(rows)
    mask_valid = ~df["dewey"].apply(is_missing_label)
    df = df[mask_valid].copy()
    df = df[df["dewey"].astype(str).str.len() > 0]
    df["dewey"] = df["dewey"].astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str)
    df["desc"]  = df["desc"].fillna("").astype(str)
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
    return X.astype(str)

# ---------------------------
# Dataset HF + split estratificado seguro
# ---------------------------

def stratified_split_with_rare(df: pd.DataFrame, test_size: float, seed: int, min_per_class: int = 2):
    y = df["dewey"].astype(str).values
    idx_all = np.arange(len(df))
    cnt = Counter(y)
    mask_common = np.array([cnt[label] >= min_per_class for label in y])
    idx_common = idx_all[mask_common]
    idx_rare   = idx_all[~mask_common]

    def resolve_n_test(ts, n):
        if 0 < ts < 1:
            return max(1, min(n-1, ceil(ts * n)))
        ts = int(ts)
        return max(1, min(n-1, ts))

    if len(idx_common) == 0 or len(set(y[idx_common])) <= 1:
        n_test = resolve_n_test(test_size, len(idx_all))
        perm = np.random.default_rng(seed).permutation(idx_all)
        idx_test = perm[:n_test]
        idx_train = perm[n_test:]
    else:
        n_test_common = resolve_n_test(test_size, len(idx_common))
        test_prop_common = n_test_common / len(idx_common)
        (idx_train_c, idx_val_c) = train_test_split(
            idx_common, test_size=test_prop_common, random_state=seed,
            stratify=y[idx_common]
        )
        idx_train = np.concatenate([idx_train_c, idx_rare], axis=0)
        idx_test  = idx_val_c

    return np.sort(idx_train), np.sort(idx_test)

# ---------------------------
# Trainer con class weights
# ---------------------------
from transformers import Trainer
import torch

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        # Movemos a device; casteamos a dtype correcto en compute_loss
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if self.class_weights is not None:
            # Match dtype & device de los logits (fp16/fp32, cpu/gpu)
            cw = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss




# ---------------------------
# Métricas
# ---------------------------

metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")
metric_bal_acc = evaluate.load("accuracy")  # balanced acc la calculamos con sklearn al final

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    out = {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }
    return out

# ---------------------------
# Main
# ---------------------------
def make_training_args(args, fp16: bool):
    desired = dict(
        output_dir=args.artifacts_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        seed=args.random_state,
        fp16=fp16,
        report_to="none",
        load_best_model_at_end=True,              # default True (se sobrescribe si pasás flag)
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
    )

    # Overrides desde CLI
    if args.evaluation_strategy is not None:
        desired["evaluation_strategy"] = args.evaluation_strategy
    if args.save_strategy is not None:
        desired["save_strategy"] = args.save_strategy
    if args.eval_steps is not None:
        desired["eval_steps"] = args.eval_steps
    if args.save_steps is not None:
        desired["save_steps"] = args.save_steps
    if args.metric_for_best_model is not None:
        desired["metric_for_best_model"] = args.metric_for_best_model
    desired["load_best_model_at_end"] = bool(args.load_best_model_at_end) or desired["load_best_model_at_end"]
    if args.warmup_ratio is not None:
        desired["warmup_ratio"] = args.warmup_ratio
    if args.warmup_steps is not None:
        desired["warmup_steps"] = args.warmup_steps
    if args.weight_decay is not None:
        desired["weight_decay"] = args.weight_decay

    if args.auto_find_batch_size:
        desired["auto_find_batch_size"] = True
    if args.gradient_accumulation_steps is not None:
        desired["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Compat con tu versión (eval_strategy/save_strategy vs evaluation_strategy)
    sig = inspect.signature(TrainingArguments.__init__).parameters
    sig_keys = set(sig.keys())

    # per_device_* -> per_gpu_* si toca
    if "per_device_train_batch_size" not in sig_keys and "per_gpu_train_batch_size" in sig_keys:
        desired["per_gpu_train_batch_size"] = desired.pop("per_device_train_batch_size")
        if "per_gpu_eval_batch_size" in sig_keys and "per_device_eval_batch_size" in desired:
            desired["per_gpu_eval_batch_size"] = desired.pop("per_device_eval_batch_size")
        else:
            desired.pop("per_device_eval_batch_size", None)

    # evaluation_strategy -> eval_strategy si tu firma lo pide
    if "evaluation_strategy" not in sig_keys and "eval_strategy" in sig_keys and "evaluation_strategy" in desired:
        desired["eval_strategy"] = desired.pop("evaluation_strategy")

    # Si pediste best model, asegura MATCH entre eval y save
    if desired.get("load_best_model_at_end"):
        es = desired.get("eval_strategy", desired.get("evaluation_strategy", None))
        if es is not None and "save_strategy" in desired:
            desired["save_strategy"] = es

    # Fallback ultra-viejo: sin claves de strategy
    if ("evaluation_strategy" not in sig_keys) and ("eval_strategy" not in sig_keys):
        desired.pop("evaluation_strategy", None)
        desired.pop("eval_strategy", None)
        if "eval_steps" in sig_keys and "eval_steps" not in desired:
            desired["eval_steps"] = 500
        if "save_steps" in sig_keys and "save_steps" not in desired:
            desired["save_steps"] = 1000
        if "do_eval" in sig_keys:
            desired["do_eval"] = True

    # Filtrado final por firma real
    filtered = {k: v for k, v in desired.items() if k in sig_keys}
    return TrainingArguments(**filtered)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--text_fields", type=str, default="both", choices=["title","desc","both"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts_hf")

    # Entrenamiento
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_per_class", type=int, default=2)

    parser.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default=None)
    parser.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)

    # Mejor checkpoint
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, choices=["accuracy", "f1_macro"], default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    # (Opcionales pero útiles en VRAM)
    parser.add_argument("--auto_find_batch_size", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    # (Opcional) logging más controlable
    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    print("Cargando datos…")
    df = load_dataset(args.data)
    if df.empty:
        raise SystemExit("No hay datos válidos.")

    X = build_text_column(df, args.text_fields)
    y = df["dewey"].astype(str).values

    # Label mapping
    labels_sorted = sorted(pd.Series(y).unique().tolist())
    label2id = {lbl:i for i,lbl in enumerate(labels_sorted)}
    id2label = {i:lbl for lbl,i in label2id.items()}
    y_ids = np.array([label2id[v] for v in y], dtype=np.int64)

    idx_train, idx_val = stratified_split_with_rare(
        df, test_size=args.test_size, seed=args.random_state, min_per_class=args.min_per_class
    )

    df_train = pd.DataFrame({"text": X.iloc[idx_train].values, "label": y_ids[idx_train]})
    df_val   = pd.DataFrame({"text": X.iloc[idx_val].values,   "label": y_ids[idx_val]})

    # Datasets HF
    ds_train = Dataset.from_pandas(df_train, preserve_index=False)
    ds_val   = Dataset.from_pandas(df_val, preserve_index=False)
    dsd = DatasetDict({"train": ds_train, "validation": ds_val})

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Clases: {len(labels_sorted)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=args.max_length
        )

    dsd = dsd.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Class weights (desbalanceo)
    counts = Counter(df_train["label"].tolist())
    num_classes = len(labels_sorted)
    weights = torch.ones(num_classes, dtype=torch.float32)
    total = len(df_train)
    for c in range(num_classes):
        freq = counts.get(c, 0)
        weights[c] = 1.0 if freq == 0 else total / (num_classes * freq)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )


    print("Transformers:", __import__("transformers").__version__)
    print("TrainingArguments accepts:", sorted(inspect.signature(TrainingArguments.__init__).parameters.keys()))


    training_args = make_training_args(args, fp16=torch.cuda.is_available())

    # Trainer con weights y early stopping
    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("Entrenando… (GPU:", torch.cuda.is_available(), ")")
    trainer.train()

    # Eval final
    eval_metrics = trainer.evaluate()
    print("Eval:", eval_metrics)

    # Guardado de artefactos HF + mapas
    trainer.save_model(args.artifacts_dir)
    tokenizer.save_pretrained(args.artifacts_dir)
    with open(os.path.join(args.artifacts_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.artifacts_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # Métricas extendidas (confusion matrix + classification_report sklearn)
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
    preds = trainer.predict(dsd["validation"])
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    rep = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(num_classes)], zero_division=0, output_dict=True)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "eval": eval_metrics,
            "balanced_accuracy": bal_acc,
            "classification_report": rep,
            "num_classes": num_classes,
            "classes": [id2label[i] for i in range(num_classes)],
            "text_fields": args.text_fields,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "patience": args.patience,
            "random_state": args.random_state
        }, f, ensure_ascii=False, indent=2)

    cm_df = pd.DataFrame(cm, index=[id2label[i] for i in range(num_classes)],
                            columns=[id2label[i] for i in range(num_classes)])
    cm_df.to_csv(os.path.join(args.artifacts_dir, "confusion_matrix.csv"), encoding="utf-8")

    print("✅ Listo. Artefactos en:", args.artifacts_dir)

if __name__ == "__main__":
    main()
