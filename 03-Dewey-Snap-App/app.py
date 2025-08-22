#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import subprocess
import sys
import os
import tempfile
import json
import uuid
import re

app = Flask(__name__)

# --- Configurables por entorno ---
PREDICT_SCRIPT = os.getenv("PREDICT_SCRIPT", "./02-Training-Testing-Model/01-predict_pdf_xlmr.py")
HF_ARTIFACTS_DIR = os.getenv("HF_ARTIFACTS_DIR", "./02-Training-Testing-Model/artifacts_hf")
DEFAULT_TEXT_FIELDS = os.getenv("TEXT_FIELDS", "both")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_MAX_LENGTH = int(os.getenv("MAX_LENGTH", "384"))
DEFAULT_BATCH_INFER = int(os.getenv("BATCH_SIZE_INFER", "8"))
DEFAULT_GENERALIZE = os.getenv("GENERALIZE", "1") not in ("0", "false", "False")

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_last_json_from_text(text: str):
    """
    Intenta recuperar el último objeto JSON de un stdout que puede contener logs.
    Estrategia: buscar el último bloque {...} balanceado.
    """
    # Tomamos la última llave de cierre y retrocedemos buscando la apertura
    last_close = text.rfind("}")
    if last_close == -1:
        return None
    # Buscamos desde el principio hasta last_close el último '{' que pueda balancear
    # Hacemos un sweep simple con un contador
    candidate = None
    stack = 0
    start_idx = None
    for i, ch in enumerate(text[:last_close+1]):
        if ch == "{":
            if stack == 0:
                start_idx = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start_idx is not None:
                candidate = text[start_idx:i+1]  # JSON potencial
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    # Validación básica
    if "file" not in request.files:
        return jsonify({"error": "No file part (campo 'file' ausente)"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Solo se aceptan archivos .pdf"}), 400

    # Parámetros opcionales por query/form (sobrescriben defaults)
    text_fields = request.form.get("text_fields", DEFAULT_TEXT_FIELDS)
    top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
    max_length = int(request.form.get("max_length", DEFAULT_MAX_LENGTH))
    batch_infer = int(request.form.get("batch_size_infer", DEFAULT_BATCH_INFER))
    generalize = request.form.get("generalize", str(int(DEFAULT_GENERALIZE))) not in ("0", "false", "False")

    artifacts_dir = request.form.get("artifacts", HF_ARTIFACTS_DIR)
    predict_script = request.form.get("script", PREDICT_SCRIPT)

    if not os.path.isfile(predict_script):
        return jsonify({"error": f"No se encuentra el script '{predict_script}'"}), 500
    if not os.path.isdir(artifacts_dir):
        return jsonify({"error": f"No se encuentra el directorio de artefactos '{artifacts_dir}'"}), 500

    # Guardar PDF en archivo temporal seguro
    safe_name = secure_filename(f.filename)
    pdf_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="upload_", dir=tempfile.gettempdir())
    pdf_path = pdf_tmp.name
    try:
        f.save(pdf_path)
    finally:
        pdf_tmp.close()

    # Ruta temporal para el JSON de salida (le decimos al script dónde escribir)
    json_tmp = os.path.join(tempfile.gettempdir(), f"report_{uuid.uuid4().hex}.json")

    # Construimos el comando (cada flag/valor en su token)
    cmd = [
        sys.executable,  # usa el mismo intérprete/entorno actual
        predict_script,
        "--pdf", pdf_path,
        "--artifacts", artifacts_dir,
        "--text_fields", text_fields,
        "--top_k", str(top_k),
        "--max_length", str(max_length),
        "--batch_size_infer", str(batch_infer),
        "--report_json", json_tmp,
    ]
    if generalize:
        cmd.append("--generalize")
    # Opcional: si querés desactivar AMP por defecto, podrías añadir "--no_amp" según form/env.

    try:
        # Timeout razonable (ajustable por env var)
        timeout_s = int(os.getenv("PREDICT_TIMEOUT", "600"))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        # Limpieza de temporales y salida
        try:
            os.remove(pdf_path)
        except Exception:
            pass
        return jsonify({"error": f"Inferencia excedió el timeout de {timeout_s}s"}), 504

    # Intentamos cargar desde el archivo JSON (fuente más confiable)
    payload = None
    if os.path.isfile(json_tmp):
        try:
            with open(json_tmp, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as e:
            payload = None

    # Si no hubo archivo, intentamos extraer JSON del stdout
    if payload is None:
        if result.returncode != 0:
            # Error del proceso: devolvemos stderr (acotado)
            err = (result.stderr or "").strip()
            out = (result.stdout or "").strip()
            # Evitamos respuestas gigantes
            return jsonify({
                "error": "Falló la clasificación",
                "returncode": result.returncode,
                "stderr": err[-4000:],  # últimos 4k chars
                "stdout_tail": out[-1000:]  # pista para debug
            }), 500
        # Proceso OK pero sin archivo: intentar parsear stdout
        payload = parse_last_json_from_text(result.stdout)
        if payload is None:
            # Último recurso: devolver tails para diagnosticar
            return jsonify({
                "error": "No se pudo obtener JSON de salida",
                "stdout_tail": (result.stdout or "")[-2000:],
                "stderr_tail": (result.stderr or "")[-1000:]
            }), 500

    # Limpieza de archivos temporales
    try:
        if os.path.isfile(pdf_path):
            os.remove(pdf_path)
    except Exception:
        pass
    try:
        if os.path.isfile(json_tmp):
            os.remove(json_tmp)
    except Exception:
        pass

    return jsonify(payload), 200


if __name__ == "__main__":
    # Puedes fijar HOST/PORT por variables de entorno
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") not in ("0", "false", "False")
    app.run(host=host, port=port, debug=debug)
