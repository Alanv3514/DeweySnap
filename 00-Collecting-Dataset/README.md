# 📚 Books-Scrapper

Aplicación desarrollada en Python para consultar dos APIs de bases de datos de libros, recolectando información relevante como:

- Título
- Clasificación Dewey
- Breve descripción *(cuando está disponible)*

---

## 🎯 Objetivo

Preparar un banco de datos para entrenar modelos de categorización automática de PDFs mediante inteligencia artificial.

---

## ⚙️ Modo de uso

- El script `DataSetScrapper.py` contiene la lógica principal del nodo recolector:
  - Implementa funciones específicas para cada API.
  - Organiza la recolección en *batches* de N intentos.
  - Por cada batch, realiza N consultas en un rango de IDs suministrado.
  - Si la información contiene **título** y **clasificación Dewey**, el intento se considera exitoso.
  - Los intentos fallidos se descartan automáticamente.
  - El algoritmo itera indefinidamente hasta recolectar al menos **100 datos válidos**.

> 💡 Puede mejorarse implementando un timeout y ajustando dinámicamente el tamaño del batch según la tasa de éxito.

- El script `multi.py` lanza `DataSetScrapper.py` en **subprocesos** para acelerar la recolección.
  - La cantidad de subprocesos y el tamaño de los batches deben ajustarse según el hardware disponible.

---

## 📈 Resultados

En poco más de **40 minutos**, se lograron recolectar **10 archivos `.txt`** con aproximadamente **100 datos válidos** cada uno. Luego fueron **combinados manualmente** en un archivo final.

### 🖥️ Hardware utilizado
- **Procesador:** Intel i5 13400F
- **Internet:** 250Mbps
