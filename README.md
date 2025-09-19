# Microproyecto - Entrega 1

**Maestría en Inteligencia Artificial**
**Clase de Desarrollo de Soluciones**

**Presentado por:**

* Braulio Martínez Reyes
* Hada Licette Sandoval Nausa
* Julio Andrés Pachón Latorre
* Sergio Alejandro Oyola Chaparro


## Descripción del proyecto

El objetivo de este proyecto es desarrollar y desplegar un producto de analítica de datos. Se debe seleccionar un problema y un conjunto de datos asociado. Se desarrolla la solución analítica desde la comprensión del problema de negocio y los datos, hasta el despliegue de la misma. Se debe considerar un problema de aprendizaje supervisado.


---


## Modelos de Ensamble – Un Proyecto de Clasificación Usando XGBoost para la Industria Hotelera

En la industria hotelera, es esencial estimar si un cliente cumplirá con una reserva. En este caso, contamos con un conjunto de datos que contiene información sobre el comportamiento de los clientes en relación con las reservas y si estas fueron efectivamente realizadas o canceladas. Este proyecto implementará XGBoost, un algoritmo de ensamble, para predecir si un cliente cancelará su reserva y clasificar el estado de la misma en función del comportamiento observado (datos etiquetados).


## Contenido

1. Objetivo  
2. Contexto del Problema  
3. Implementación del Modelo  
    * a. Importación de librerías  
    * b. Lectura y análisis de datos  
    * c. División de datos  
    * d. Modelo base  
    * e. XGBoost y el número óptimo de estimadores  
    * f. Entrenamiento y resultados del mejor modelo  
    * g. Análisis de características  
4. Conclusiones  
5. Comentarios finales y pasos siguientes  


## Descripción del Conjunto de Datos y Diccionario

El conjunto de datos contiene diversos atributos relacionados con los detalles de las reservas de los clientes. A continuación, se presenta un diccionario de datos detallado:

| **Atributo**                      | **Descripción**                                                                                                                            |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **Booking_ID**                    | Identificador único de cada reserva                                                                                                        |
| **No of adults**                  | Número de adultos                                                                                                                          |
| **No of children**                | Número de niños                                                                                                                            |
| **noofweekend_nights**            | Número de noches de fin de semana (sábado o domingo) que el huésped se hospedó o reservó en el hotel                                       |
| **noofweek_nights**               | Número de noches entre semana (lunes a viernes) que el huésped se hospedó o reservó en el hotel                                            |
| **typeofmeal_plan**               | Tipo de plan de comidas reservado por el cliente                                                                                           |
| **requiredcarparking_space**      | Indica si el cliente requiere espacio de parqueo (0 - No, 1 - Sí)                                                                           |
| **roomtypereserved**              | Tipo de habitación reservada por el cliente (cifrada/codificada por INN Hotels)                                                             |
| **lead_time**                     | Número de días entre la fecha de reserva y la fecha de llegada                                                                             |
| **arrival_year**                  | Año de la fecha de llegada                                                                                                                  |
| **arrival_month**                 | Mes de la fecha de llegada                                                                                                                  |
| **arrival_date**                  | Día del mes de la fecha de llegada                                                                                                          |
| **Market segment type**           | Designación del segmento de mercado                                                                                                         |
| **repeated_guest**                | Indica si el cliente es un huésped repetido (0 - No, 1 - Sí)                                                                                |
| **noofprevious_cancellations**    | Número de reservas previas canceladas por el cliente                                                                                        |
| **noofpreviousbookingsnot_canceled** | Número de reservas previas no canceladas por el cliente                                                                                  |
| **avgpriceper_room**              | Precio promedio por día de la reserva; los precios de las habitaciones son dinámicos (en euros)                                             |
| **noofspecial_requests**          | Número total de solicitudes especiales realizadas por el cliente (ej. piso alto, vista específica, etc.)                                    |
| **booking_status**                | Indicador de si la reserva fue cancelada o no                                                                                               |


## Agradecimientos

Este proyecto se basa en un ejercicio original realizado durante mi Maestría en Inteligencia Artificial. El conjunto de datos utilizado en este proyecto fue obtenido de: https://bit.ly/4e3bnQD


## Teoría del Modelo

Este proyecto implementa un modelo de **Aprendizaje Supervisado** utilizando el algoritmo **XGBoost** para clasificar la variable objetivo.

XGBoost (eXtreme Gradient Boosting) es un algoritmo avanzado de aprendizaje automático que mejora la precisión de predicción al combinar múltiples modelos más débiles para formar un ensamble robusto y optimizado. Construye árboles de decisión como modelos base y aplica técnicas de regularización y optimización durante la construcción de los árboles para reducir el sobreajuste y mejorar la generalización.  

Algunos parámetros clave del modelo XGBoost son:

- **max_depth**:  
   Controla la profundidad máxima de cada árbol de decisión. Limitar la profundidad ayuda a prevenir el sobreajuste al reducir la complejidad del modelo.

- **n_estimators**:  
   Especifica el número de árboles a construir en el ensamble. Cada árbol se añade secuencialmente, contribuyendo a la predicción final. Incrementar el número de estimadores generalmente mejora la precisión, pero también aumenta los costos computacionales.


## Resultados

El modelo mejorado de XGBoost con 45 estimadores alcanzó una precisión del 82%, reflejando un desempeño general sólido. Sin embargo, el foco principal está en su capacidad para identificar correctamente las cancelaciones (Clase 1), las cuales son críticas para la toma de decisiones en el negocio.  

El recall del 74% para la Clase 1 indica que el modelo logra identificar la mayoría de las cancelaciones, aunque aún existe espacio para mejorar. Mantener un balance entre precisión y recall asegura menos falsos positivos al mismo tiempo que captura una gran parte de las cancelaciones—lo cual es crucial para estrategias enfocadas en reducir las interrupciones en las reservas.