# Titanic Survival Prediction - Challenge

## Descripción del Proyecto

Este proyecto tiene como objetivo predecir la supervivencia de los pasajeros del Titanic utilizando un conjunto de datos con características como la edad, el sexo, la clase, entre otras. Para ello, se entrenaron varios modelos de clasificación, y después de una serie de pruebas, se eligió el **Random Forest Classifier** como el mejor modelo para esta tarea. El archivo resultante es un CSV con la predicción de si un pasajero sobrevivió o no, usando el `PassengerId` como identificador y la predicción de supervivencia como la etiqueta.

## Diccionario de Datos

### Conjunto de Entrenamiento

El conjunto de datos utilizado para el entrenamiento contiene las siguientes columnas:

- **PassengerId**: Identificador único para cada pasajero.
- **Pclass**: Clase del pasajero (1 = primera clase, 2 = segunda clase, 3 = tercera clase).
- **Name**: Nombre del pasajero.
- **Sex**: Género del pasajero (male = masculino, female = femenino).
- **Age**: Edad del pasajero en años.
- **SibSp**: Número de hermanos o cónyuges a bordo.
- **Parch**: Número de padres o hijos a bordo.
- **Fare**: Tarifa que pagó el pasajero.
- **Embarked**: Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton).
- **Survived**: Variable objetivo (0 = no sobrevivió, 1 = sobrevivió).

### Conjunto de Test

El conjunto de prueba contiene las mismas columnas, excepto la columna **Survived**, que es la que se predice. El archivo de test contiene únicamente las características del pasajero, y el objetivo es predecir la supervivencia basada en estas características.

## Elección del Modelo

En un principio, se probaron varios modelos de clasificación para predecir la supervivencia de los pasajeros. Se experimentaron con modelos como **Logistic Regression**, **Support Vector Machine (SVM)**, y **K-Nearest Neighbors (KNN)**. Sin embargo, el rendimiento de estos modelos no fue lo suficientemente bueno en términos de precisión, lo que llevó a explorar más modelos complejos.

Finalmente, se decidió utilizar el **Random Forest Classifier**, que ha demostrado ser muy efectivo en tareas de clasificación debido a su capacidad para manejar datos complejos y no lineales, como es el caso de este conjunto de datos. El Random Forest crea múltiples árboles de decisión y hace un promedio de las predicciones de estos árboles, lo que mejora la generalización y reduce el sobreajuste.

### Selección de Hiperparámetros

Para elegir los mejores parámetros para el modelo Random Forest, se utilizó un enfoque de búsqueda aleatoria con el método **RandomizedSearchCV**. Se seleccionaron dos hiperparámetros clave:

- **n_estimators**: El número de árboles en el bosque. Un mayor número de árboles puede mejorar el rendimiento del modelo, pero a la vez incrementa el tiempo de entrenamiento.
- **max_depth**: La profundidad máxima de cada árbol. Al limitar la profundidad de los árboles, se controla el sobreajuste, ya que los árboles muy profundos tienden a adaptarse demasiado a los datos de entrenamiento.

Se utilizaron distribuciones aleatorias para estos parámetros dentro de un rango de valores razonable (por ejemplo, entre 60 y 100 para `n_estimators` y entre 1 y 10 para `max_depth`). La búsqueda aleatoria permite explorar una gran cantidad de combinaciones sin tener que probar todas las posibles, lo que es más eficiente.

### Resultados

El mejor modelo fue encontrado con los siguientes parámetros:

- **n_estimators**: 98
- **max_depth**: 5

Este modelo se entrenó con estos parámetros y se utilizó para predecir las supervivencias en el conjunto de prueba.

## Visualización de los Datos

Antes de entrenar el modelo, se realizó una exploración y visualización de los datos para entender mejor la distribución de las características y cómo se relacionan con la variable objetivo, **Survived**.

- **Distribución de la edad**: Se generó un histograma para visualizar la distribución de las edades de los pasajeros. Se observó que muchos pasajeros eran jóvenes o adultos jóvenes.
- **Supervivencia por clase**: Se creó un gráfico de barras para mostrar la tasa de supervivencia en cada clase (Pclass). La primera clase tenía una tasa de supervivencia mucho más alta que las clases inferiores.
- **Supervivencia por sexo**: Se visualizó la supervivencia en función del sexo. Las mujeres tuvieron una tasa de supervivencia significativamente mayor que los hombres.

Estas visualizaciones ayudaron a obtener intuiciones importantes sobre cómo las características afectaban la supervivencia y ayudaron a guiar la selección de los mejores modelos.

## Cómo Ejecutar el Proyecto

### Requisitos

- Python 3.x
- Pandas
- NumPy
- Matplotlib (para visualización)
- Seaborn (para visualización)
- Scikit-learn

### Instrucciones

1. Clona este repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el script para entrenar el modelo y realizar las predicciones:
   ```bash
   python titanic_model.py
   ```
4. El archivo de predicción se generará como `titanic_predictions.csv`, que contiene las predicciones de supervivencia de los pasajeros del conjunto de prueba.

## Conclusión

Este proyecto muestra cómo la elección del modelo adecuado y la sintonización de sus hiperparámetros pueden mejorar significativamente el rendimiento en un problema de clasificación. Aunque inicialmente probamos con varios modelos, el Random Forest fue el más adecuado para este conjunto de datos, brindando un rendimiento robusto y preciso. 
