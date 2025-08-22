# 01-Training-Testing-Model
Aplicación para entrenamiento y testeo del modelo a utilizar.

## Objetivo
Entrenar un modelo de categorización automática de libros en base al dataset recolectado en la etapa anterior.

## Modo de uso
El script principal debe cargar el dataset, dividirlo en conjuntos de entrenamiento y prueba, entrenar el modelo y evaluar su desempeño. Se recomienda guardar el modelo entrenado para su uso posterior.

## Archivos esperados
- 00-train_* son archivos de entrenamiento, la ultima version es 00-train_dewey_xlmr.py 
comando de entrenamiento utilizado:

conda run --live-stream --name torch129 python 00-train_dewey_xlmr.py `  --data ./libros.txt `  --text_fields both `  --batch_size 16 `  --gradient_accumulation_steps 2 `  --epochs 10 `  --lr 2e-5 `  --patience 3 `  --max_length 384 `  --artifacts_dir ./artifacts_hf `  --evaluation_strategy epoch `  --save_strategy epoch ` --gradient_accumulation_steps 2 --auto_find_batch_size --load_best_model_at_end `  --metric_for_best_model f1_macro `  --warmup_ratio 0.1 `  --weight_decay 0.01

- 01-predict_* son archivos de prediccion, la ultima version es 01-predict_pdf_xlmr.py 
comando utilizado:
conda run --live-stream --name torch129 python ./01-predict_pdf_xlmr.py `  --pdf ./libro.pdf `  --artifacts ./artifacts_hf `  --text_fields both `  --top_k 5 `  
--generalize `  --max_length 384 `  --batch_size_infer 8
      