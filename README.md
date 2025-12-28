# Segmentación de lesiones nuevas o cambiantes a través de MRI usando la red CNN nnU-Net v2

## Introducción
Este repositorio contiene información sobre el desarrollo del proyecto (TFM: "Detección de lesiones nuevas o cambiantes en EM") de segmentación longitudinal de lesiones producidas por la Esclerosis Múltiple, usando la red CNN **nnU-Net v2**.
Las imágenes contienen las etiquetas:
- 0: zona sin lesión
- 1: lesión estable
- 2: lesión nueva

## Pipeline del proyecto
<img width="1978" height="1066" alt="pipeline_4" src="https://github.com/user-attachments/assets/97669c36-7968-41fd-b7fe-908f68ac8d1f" />


## Contenido del repositorio

Se incorporan el código para realizar las siguientes tareas:

- Análisis de los datasets empleados ImaginEM (para entrenamiento) y para MSSEG2
- Separación del dataset ImaginEM en 70% train, 15% validation y 15% test para aplicar la técnica holdout en el entrenamiento de los modelos. (Cross-validation descartada por limitaciones de computaión)
- Preprocesamiento.
- Configuraciones personalizadas de los entrenamientos: 20 épocas, 50 épocas, 100 épocas y 250 épocas. 
- Evaluación en ImaginEM y en MSSEG2.
- Postprocesamiento: estudio de distintas técnicas (filtrado por volumen y por homología persistente) para reducir falsos positivos mejorando las métricas
- Resultados y métricas finales.

El código se encuentra en la carpeta "code" del repositorio. Ahí están los ficheros:
- 01_analisis_estadistico_ImaginEM_MSSEG2.py--> código para realizar el estudio estadístico de los datasets ImaginEM y MSSEG2 de la memoria.
- 02_HOLDOUT.py--> código para realizar la división 70% train, 15% validation y 15% test sobre ImaginEM, usando los canales FLAIR baseline y FLAIR followup.
- 03_custom_train.py--> código para realizar los  entrenamientos personalizados: 20 épocas, (ya viene de base en nnU-Net v2), 50 épocas, 100 épocas y 250 épocas.
- 04_predicciones_Eval_test_split_MSSEG2.py--> código para realizar las predicciones de los 4 modelos entrenados en el test-split de ImaginEM y en MSSEG2.
- 05_metrics_custom_train_ImaginEM_MSSEG2.py--> código para obtener las métricas de estos 4 modelos en ImaginEM y en MSSEG2, a nivel voxel-wise, lesion-wise e ID-wise.
- 06_postprocess_evaluation.py--> código para realizar los 4 postprocesamientos distintos (primero aplicando filtros de volumen en $mm^{3}$, luego aplicando técnicas basadas en la homología persistente de dimensión 0, $H_{0}$

##  Datos utilizados

- Dataset cedido por el grupo ImaginEM y el Hospital Clínic de Barcelona de 349 personas con Esclerosis Múltiple. Presenta las secuencias:
  - FLAIR baseline (primera imagen de nuestro dataset dado un ID)
  - FLAIR follow-up (imagen posterior dado un ID (tomada entre 1 y 3 después de la baseline)
  - Segmentaciones manuales de las lesiones, realizadas por los radiólogos especialistas

-Distribución de la muestra: 
  - La cohorte presenta un rango de edad de 50+-10 años
  - Está comppuesta aproximadamente por un 60/70% de mujeres.

- Dataset externo para evaluación:
  - MSSEG-2 (únicamente tiene FLAIR baseline y FLAIR followup), es importante realizar esta evaluación externa para evaluar el grado de generalización que tiene el modelo, y también si se ve afectado por la diferencia de escáneres usados para tomar esas resonancias.


##  Herramientas empleadas

- Google Colab Pro+
- Google Drive
- Python
- nnU-Net v2
- PyTorch
- nibabel
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- giotto-tda
- CubicalPersistence
  

##  Objetivo

Desarrollar un modelo basado en la nnU-Net v2, que sea capaz de distinguir entre zonas sin lesiones, zonas con lesiones estables y zonas con lesiones nuevas. Se evalúa su rendimiento mediante un dataset externo, MSSEG2.

## Licencia

[![CC BY-NC-SA 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Attribution-NonCommercial-ShareAlike 4.0 International


---

 Autora: Sofía Ramírez López  
 Trabajo Fin de Máster: Detección de lesiones nuevas o cambiantes en EM
