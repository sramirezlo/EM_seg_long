# Segmentación de lesiones nuevas o cambiantes a través de MRI usando la red CNN nnU-Net v2

## Introducción
Este repositorio contiene información sobre el desarrollo del proyecto (TFM: "Detección de lesiones nuevas o cambiantes en EM") de segmentación longitudinal de lesiones producidas por la Esclerosis Múltiple, usando la red CNN **nnU-Net v2**.
Las imágenes contendrán las etiquetas:
- 0: zona sin lesión
- 1: lesión estable
- 2: lesión nueva

## Contenido del repositorio

Se incorporan el código para realizar las siguientes tareas:

- Análisis de los datasets empleados ImaginEM (para entrenamiento) y para MSSEG2.
- Separación del dataset ImaginEM en 70% train, 15% validation y 15% test para aplicar la técnica holdout en el entrenamiento de los modelos. (Cross-validation descartada por limitaciones de computaión)
- Preprocesamiento.
- Configuraciones personalizadas de los entrenamientos: 20 épocas, 50 épocas, 100 épocas y 250 épocas. 
- Evaluación en ImaginEM y en MSSEG2.
- Postprocesamiento: estudio de distintas técnicas (filtrado por volumen y por homología persistente) para reducir falsos positivos mejorando las métricas
- Resultados y métricas finales.

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

- nnU-Net v2
- Python
- PyTorch
- Google Colab Pro+

##  Objetivo

Desarrollar un modelo basado en la nnU-Net v2, que sea capaz de distinguir entre zonas sin lesiones, zonas con lesiones estables y zonas con lesiones nuevas. Se evalúa su rendimiento mediante un dataset externo, MSSEG2.

## Licencia

[![CC BY-NC-SA 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Attribution-NonCommercial-ShareAlike 4.0 International


---

 Autora: Sofía Ramírez López  
 Trabajo Fin de Máster: Detección de lesiones nuevas o cambiantes en EM
