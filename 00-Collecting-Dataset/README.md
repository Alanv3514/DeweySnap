# Books-Scrapper
Una aplicacion desarrollada en python con el fin de consultar 2 apis de databases de libros, recolectar informacion relevante como titulo, categoria dewey y una breve descripcion
(descripcion no siempre encuentra)

## Objetivo
Preparar un baco de datasets para posteriormente entrenar modelos de categorizacion automatica a travez de ia para PDFs

## Modo de uso
El script DataSetScrapper.py contiene la logica principal del nodo recolector, con funciones para cada api y organizando la recoleccion en batchs de N intentos, por cada batch busca hacer N consultas a las api en un rango de ids suministrado, si la informacion contiene titulo y dewey el intento es exitoso y se computa para el resultado final, caso contrario se descarta. El algoritmo itera indefinidamente hasta alcanzar 100 o mas datos validos. Puede mejorarse seteando un timeout y un batch dinamico en funcion de la velocidad de obtencion de exitos.
El script multi.py lanza el DataSetScrapper.py en subprocesos para poder disminuir el tiempo de recoleccion lo mayor posible. La cantidad de subprocesos y el tamaño de los batch de datos deben experimentarse segun su hardware.

##Resultados
En poco mas de 40 minutos se pudo completar la tarea de conseguir 10 txt con aproximadamente 100 datos validos cada uno que luego fueron mergeados manualmente a un txt final.
###Procesador : Intel i5 13400f

