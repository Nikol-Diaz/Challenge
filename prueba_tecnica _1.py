import pandas as pd
from IPython.display import Markdown, display

# Importacion de Datos
# cargar csv
Train_data = pd.read_csv('train.csv', encoding='utf-8')
Test_data = pd.read_csv('test.csv', encoding='utf-8')

# Mostrar las primeras filas
display(Markdown("## Visualización de las primeras filas del Train_dataset"))
print(Train_data.head())

display(Markdown("## Visualización de las primeras filas del Test_dataset"))
print(Test_data.head())

# Información general del dataset
display(Markdown("## Informacion del  Train_dataset"))
print(Train_data.info())


display(Markdown("## Informacion del  Test_dataset"))
print(Test_data.info())

# Resumen estadístico
display(Markdown("## Resumen estadístico del Train_dataset"))
print(Train_data.describe())


# Resumen estadístico
display(Markdown("## Resumen estadístico del Test_dataset"))
print(Test_data.describe())


display(Markdown("### Dimensiones del Train_dataset"))
print(Train_data.shape)

display(Markdown("### Dimensiones del Test_dataset"))
print(Test_data.shape)


# Limopieza de datos

display(Markdown("## Valores nulos en el Train_dataset"))
print(Train_data.isnull().sum())


display(Markdown("## Valores nulos en el Test_dataset"))
print(Test_data.isnull().sum())

#cabins tiene demasiados valores nulos, por lo que se elimina la columna
display(Markdown("### Eliminación de la columna Cabin"))
Train_data = Train_data.drop(columns=['Cabin'])
Test_data = Test_data.drop(columns=['Cabin'])


# Ya que son solo 2 valores nulos, lo reemplazamos por la moda (o valor que mas se repite)
display(Markdown("### Reemplazo de valores nulos en la columna Embarked con la moda"))
Embarked_mode = Train_data["Embarked"].mode()[0]
Train_data["Embarked"].fillna(Embarked_mode, inplace=True)


display(Markdown("### Verificar los valores unicos de la columna Embarked"))
print(Train_data['Embarked'].unique())

display(Markdown("### Verificar los valores unicos de la columna Sex"))
print(Train_data['Sex'].unique())


display(Markdown("### Dividir la columna Embarked en varias columnas"))
Train_data= pd.get_dummies(Train_data, columns=["Embarked"], drop_first=True)
Test_data = pd.get_dummies(Test_data, columns=["Embarked"], drop_first=True)
print("columnas nuevas: Embarked_Q, Embarked_S " )

display(Markdown("### Dividir la columna Sex en varias columnas"))
Train_data= pd.get_dummies(Train_data, columns=["Sex"], drop_first=True)
Test_data = pd.get_dummies(Test_data, columns=["Sex"], drop_first=True)
print("columnas nuevas: Sex_male " )



#Tenemos 177 valores nulos en la columna Age
print("Pasajeros en primera clase con Age NULL", Train_data.loc[(Train_data['Age'].isnull()) & (Train_data['Pclass'] == 1)].shape[0])
print("Pasajeros en segunda clase con Age NULL", Train_data.loc[(Train_data['Age'].isnull()) & (Train_data['Pclass'] == 2)].shape[0])
print("Pasajeros en tercera clase con Age NULL", Train_data.loc[(Train_data['Age'].isnull()) & (Train_data['Pclass'] == 3)].shape[0])


display(Markdown("### Reemplazo de valores nulos en la columna Age con la mediana"))
Train_data["Age"] = Train_data.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))


display(Markdown("### Eliminación de las columnas Name y Ticket por no ser relevantes"))
Train_data = Train_data.drop(columns=["Name", "Ticket"])
Test_data = Test_data.drop(columns=["Name", "Ticket"])

display(Markdown("### Visualizar nuevamente información del dataset"))
print(Train_data.info())


#Visualizar los valores unicos de la columna
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma de una variable numérica (modifica 'nombre_columna')
# EN esta visualiaación se muestra la distribución de la variable Age destancando que la mayoria de los pasajeros tienen entre 20 y 25 años
display(Markdown("### Histograma de la variable Age- Imagen guardada como Age.png"))
plt.figure(figsize=(8, 4))
sns.histplot(Train_data['Age'], bins=15,binwidth=5, kde=True, color='Green')

plt.title('Distribución de Edad')
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.savefig('Age.png')


# Aqui veo la cantidad de pasajeros que sobrevivieron y los que no sobrevivieron por edad
# se puede destacar que la mayoria de los pasajeros que no sobrevivieron tenian entre 20 y 30 años

display(Markdown("### Histograma de la variable Age con la variable Survived- Image guardada como Age_Survived.png"))
plt.figure(figsize=(6, 4))
sns.histplot(data=Train_data, x="Age", hue="Survived", bins=30, kde=True)  # Corrección del nombre y mejoras
plt.title("Supervivencia por Edad")
plt.xlabel("Edad")
plt.ylabel("Cantidad")
plt.savefig('Age_Survived.png')



# Boxplot para detectar outliers
# siguiendo la logica de los tickets para 1ra clase siempre sera mas caro que para 2da y 3ra clase podemos ver que incluso en la 1ra clase hay valores atipicos
display(Markdown("### Boxplot de la variable Fare por Clase - Imagen guardada como Fare.png"))
plt.figure(figsize=(8, 5))
sns.boxplot(x="Pclass", y="Fare", data=Train_data)
plt.title("Distribución de la Tarifa por Clase")
plt.xlabel("Clase")
plt.ylabel("Tarifa")
plt.savefig('Fare.png')

# Conteo de sobrevivientes por género, podemos visualizar que la mayoria de los sobrevivientes son mujeres
display(Markdown("### Supervivencia por Género imagen guardada como Survived.png"))
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", hue="Sex_male", data=Train_data)
plt.title("Supervivencia por Género")
plt.xlabel("Sobrevivió (0 = No, 1 = Sí)")
plt.ylabel("Cantidad")
plt.legend(["Femenino", "Masculino"], title="Género")
plt.savefig('Survived.png')



# Modelo de Machine Learning
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


# como ya tenemos dos dataset uno para testeo y otro training debemos dividir el dataset de training en dos partes para poder evaluar el modelo 
# ya que no tenemos la target en el dataset de testeo
X_train , X_test, Y_train, y_test = train_test_split(
    Train_data.drop(columns=["Survived"]),  # Target
    Train_data["Survived"],  # Target
    test_size=0.2,  # 20% para validación
    random_state=42
)




# Relizamos una eleccion de modelos
# Inicializar los modelos
models = {
    "Logistic Regression": LogisticRegression(),
    "Ridge Classifier": RidgeClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
}

# Vemos que el mejor modelo es el Random Forest
for name, model in models.items():
    model.fit(X_train, Y_train)
    print(f"{name}: {model.score(X_test, y_test):.4f}")


# Busqueda de hiperparametros
param_dist = {'n_estimators': randint(60,100),
              'max_depth': randint(1,10)}

# crear un objeto RandomForest
rf = RandomForestClassifier()

# Usamos RandomizedSearchCV para buscar los mejores hiperparametros
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Entrenamos el modelo
rand_search.fit(X_train, Y_train)

# Creamos la variable best_rf para guardar el mejor modelo
best_rf = rand_search.best_estimator_

# Imprimimos los mejores hiperparametros
print('Best hyperparameters:',  rand_search.best_params_)


# Entrenamos con el mejor parametro
best_rf.fit(X_train, Y_train)

# con los mejores hiperparametros entrenamos el modelo
y_pred = best_rf.predict(X_test)

# Creamos la matriz de confusion
cm = confusion_matrix(y_test, y_pred)


# Mostramos la matriz de confusion
# Vemos que el modelo detecta mejor la posibilidad de no sobrevivir que la de sobrevivir
print(cm)

# ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Mostramos el reporte de clasificación

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# Encotramos que las variables mas importantes
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

#Visualizamos las variables mas importantes
feature_importances.plot.bar()

# # Eliminamos las columnas que no son relevantes
X_train = X_train.drop(columns=["Embarked_Q", "Embarked_S"])
X_test = X_test.drop(columns=["Embarked_Q", "Embarked_S"])


#volvemos a entrenar el modelo
best_rf.fit(X_train, Y_train)



y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# Predecir las etiquetas (sobrevivió o no sobrevivió) para el conjunto de prueba
Test_data = Test_data.drop(columns=["Embarked_Q", "Embarked_S"])
predictions = best_rf.predict(Test_data)  # Usar el mejor modelo encontrado


# Crear un DataFrame con los resultados de la predicción
results = pd.DataFrame({
    'PassengerId': Test_data['PassengerId'],  # Asegúrate de que 'PassengerId' esté en los datos de prueba
    'Survived': predictions
})

# Guardar los resultados en un archivo CSV
results.to_csv('titanic_predictions.csv', index=False)




