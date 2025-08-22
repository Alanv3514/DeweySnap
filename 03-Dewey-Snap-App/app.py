from flask import Flask, request, jsonify
import subprocess
import os
import tempfile
import json

app = Flask(__name__)

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Guardar el archivo PDF en una ubicación temporal
    temp_dir = tempfile.gettempdir()
    temp_pdf_path = os.path.join(temp_dir, file.filename)
    file.save(temp_pdf_path)

    # Llamar al script predict_pdf.py para procesar el PDF y generar un reporte JSON
    result = subprocess.run(['python', './02-Training-Testing-Model/01-predict_pdf.py', '--pdf', temp_pdf_path, '--artifacts', './02-Training-Testing-Model/artifacts_ml'], capture_output=True, text=True)
    print(result.stdout)
    
    # Capturar la salida del comando y manejarla adecuadamente
    if result.returncode != 0:
        return jsonify({"error": f"Command failed with output: {result.stderr}"}), 500
    
    try:
        stdout_json = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to decode JSON from process output: {str(e)}"}), 500
    
    # Devolver el resultado como un objeto JSON en lugar de una cadena
    return jsonify(stdout_json)

if __name__ == '__main__':
    app.run(debug=True)