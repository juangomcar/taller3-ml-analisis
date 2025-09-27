#!/usr/bin/env python
# coding: utf-8

# ### PCA in Machine Learning Workflows
# #### Machine Learning I - Maestría en Analítica Aplicada
# #### Universidad de la Sabana
# #### Prof: Hugo Franco
# #### Example: Principal Component Analysis
# 
# <img src="culmen_depth.png" width="50%">

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer       

# Load the penguins dataset
penguins = sns.load_dataset('penguins')

# Display initial information
print("Dataset Overview:")
print(penguins.info())
print("\nClass Distribution:")
print(penguins['species'].value_counts())


# #### Imputation strategies implemented in SimpleImputer 
# * mean (default for numeric data)
# * median (usually more robust than mean)
# * most_frequent
# * constant (requires the filling value)

# In[2]:


# Define feature groups
numeric_features = ['bill_length_mm', 'bill_depth_mm', 
                   'flipper_length_mm', 'body_mass_g']
categorical_features = ['sex', 'island']

# Create the numeric transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create the categorical transformer with one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(
        drop='first',  # Drop first category to avoid multicollinearity
        sparse_output=False,  # Return dense array instead of sparse matrix
        handle_unknown='ignore'  # Handle new categories in test data
    ))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Print encoded feature names
def get_feature_names(preprocessor):
    # Get feature names from numeric features
    numeric_features_out = numeric_features

    # Get feature names from categorical features after encoding
    cat_features = (preprocessor
                   .named_transformers_['cat']
                   .named_steps['onehot']
                   .get_feature_names_out(categorical_features))
    
    # Combine both feature sets
    return numeric_features_out + list(cat_features)


# In[3]:


# Create full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ))
])

# Prepare data
X = penguins.drop(['species'], axis=1)
y = penguins['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# In[4]:


# Train model
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Print performance metrics
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=penguins['species'].unique(),
            yticklabels=penguins['species'].unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Get feature names after encoding
feature_names = get_feature_names(preprocessor)
print("\nEncoded Feature Names:")
print(feature_names)


# A (risky) method to deal with potential outliers: quartile capping.

# In[5]:


def cap_outliers(df, columns, lower_percentile=1, upper_percentile=99):
    df_capped = df.copy()
    for column in columns:
        lower = np.percentile(df[column].dropna(), lower_percentile)
        upper = np.percentile(df[column].dropna(), upper_percentile)
        df_capped[column] = df_capped[column].clip(lower=lower, upper=upper)
    return df_capped

# Get numerical columns from features only (excluding target)
numerical_cols_features = X.select_dtypes(include=['float64', 'int64']).columns

# Apply outlier capping
X_capped = cap_outliers(X, numerical_cols_features)

# Create pipeline with outlier capping and proper preprocessing
capped_pipe = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', KNNImputer(n_neighbors=5))
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            ))
        ]), categorical_features)
    ])),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Evaluate with outlier capping
X_train_capped, X_test_capped, y_train, y_test = train_test_split(
    X_capped, y, test_size=0.2, random_state=42)

capped_pipe.fit(X_train_capped, y_train)
y_pred_capped = capped_pipe.predict(X_test_capped)

# Evaluate results with visualization
print("\nResults with Outlier Capping:")
print("Accuracy:", accuracy_score(y_test, y_pred_capped))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_capped, zero_division=0))

# Visualize confusion matrix for capped results
plt.figure(figsize=(8, 6))
cm_capped = confusion_matrix(y_test, y_pred_capped)
sns.heatmap(cm_capped, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - With Outlier Capping')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# #### Challenge (Workshop)
# 1. Use the following code stub to perform the same task on the Cleveland Heart Disease dataset. Test the impact of each imputation strategy in the model performance. 
# 2. Compare the performance of Random Forests vs. XGBoost
# 

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=columns, na_values='?')


# # Challenge

# ## 1. Carga de Datos
# En esta sección cargamos el dataset de Cleveland desde la librería UCI.  
# Convertimos los valores faltantes representados con `?` a `NaN` para poder imputarlos después.
# 

# In[7]:


import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
           'restecg', 'thalach', 'exang', 'oldpeak', 
           'slope', 'ca', 'thal', 'target']

df = pd.read_csv(url, names=columns, na_values='?')

print(df.head())
print(df.info())


# ## Preprocesamiento

# ### 2.1 Revisión de valores faltantes
# 

# In[8]:


df.isnull().sum()


# ### 2.2 Separación de variables
# 
# En esta etapa dividimos el dataset en:
# 
# - **Features (X):** todas las variables independientes que contienen información clínica del paciente (edad, presión arterial, colesterol, etc.).  
#   Estas serán las **entradas** del modelo.
# 
# - **Target (y):** la variable objetivo, que indica la presencia o ausencia de enfermedad.  
#   Para simplificar el problema, convertimos esta variable en **binaria**:  
#   - 0 → paciente sano  
#   - 1 → paciente con enfermedad
# 
# De esta forma, el modelo aprende a clasificar entre dos clases: **sano** vs **enfermo**.
# 

# In[9]:


X = df.drop("target", axis=1)
y = df["target"].apply(lambda x: 1 if x > 0 else 0) 


# ### 2.3 Manejo de valores faltantes
# 
# 
# En el dataset, las columnas `ca` y `thal` presentan algunos valores nulos (`NaN`).  
# En lugar de eliminar registros, se aplican distintas **estrategias de imputación** para reemplazar estos valores:
# 
# - **SimpleImputer** con estrategias:
#   - mean → sustituye por la media de la columna.
#   - median → sustituye por la mediana.
#   - most_frequent → sustituye por el valor más frecuente.
#   - constant → sustituye por un valor fijo (en este caso, 0).
# 
# - **KNNImputer**:
#   - Utiliza la similitud con los registros más cercanos (*k=5 vecinos*) para estimar el valor faltante.
# 
# Cada estrategia se evalúa dentro de un **pipeline** que incluye también estandarización (`StandardScaler`) y el clasificador (`RandomForestClassifier`).  
# De esta forma, podemos medir cómo influye el método de imputación en el rendimiento del modelo.
# 
# 

# In[10]:


# Separar features y target
X = df.drop("target", axis=1)
y = df["target"].apply(lambda x: 1 if x > 0 else 0) 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

strategies = ['mean', 'median', 'most_frequent', 'constant']
results = {}

for strat in strategies:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=strat, fill_value=0)),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[strat] = accuracy_score(y_test, y_pred)

# KNNImputer
pipe_knn = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])
pipe_knn.fit(X_train, y_train)
y_pred_knn = pipe_knn.predict(X_test)
results['knn'] = accuracy_score(y_test, y_pred_knn)

results


# ## 3. Resultados de imputación

# In[11]:


pd.DataFrame(results, index=["Accuracy"]).T


# ## Discusión

# Tras aplicar las diferentes estrategias de imputación a las columnas con valores faltantes (`ca` y `thal`), se evaluó el desempeño del modelo **Random Forest** en el conjunto de prueba.  
# 
# Los resultados muestran que:
# 
# - El imputador **constant** alcanzó el mejor desempeño con ~90% de accuracy.  
# - Las estrategias **median** y **most_frequent** también funcionaron bien (~88%).  
# - **KNN** y **mean** estuvieron un poco por debajo (~87%).  
# 
# Esto indica que, aunque el dataset es relativamente robusto a la imputación, la elección del método puede generar diferencias en el rendimiento final.  
# En este caso, **constant** fue la mejor opción.
# 

# ## 4. Comparación de Modelos: Random Forest vs XGBoost
# 
# En esta sección se utilizarán:  
# - **Random Forest Classifier**  
# - **XGBoost Classifier**
# 
# Dado que en el análisis anterior se identificó que el método de imputación **constant** ofreció el mejor desempeño,  
# este se fija dentro del pipeline de ambos modelos.  
# 
# El flujo de procesamiento es el siguiente:
# 1. **Imputación:** se reemplazan los valores faltantes (`NaN`) de `ca` y `thal` por un valor constante (0).  
# 2. **Estandarización:** se escalan las variables numéricas con `StandardScaler` para homogenizar las magnitudes.  
# 3. **Entrenamiento del modelo:** se entrena cada clasificador con los datos de entrenamiento.  
# 4. **Evaluación:** se calculan métricas de desempeño (`accuracy`, `precision`, `recall`, `f1-score`) y se grafican las matrices de confusión.
# 
# De esta forma, podemos comparar directamente cuál de los dos modelos obtiene mejor rendimiento en la predicción de la enfermedad cardíaca.
# 

# In[12]:


from xgboost import XGBClassifier

# Random Forest con imputación constant
pipe_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

# XGBoost con imputación constant
pipe_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

models = {"Random Forest": pipe_rf, "XGBoost": pipe_xgb}
model_results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_results[name] = acc
    
    print(f"\n{name} Results:")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

model_results


# ## 5. Resultados y Discusión

# In[13]:


# Resultados de imputación
imputation_results = {
    "Estrategia": ["Mean", "Median", "Most Frequent", "Constant", "KNN"],
    "Accuracy": [0.8688, 0.8852, 0.8852, 0.9016, 0.8688]
}

df_imputations = pd.DataFrame(imputation_results)
print(df_imputations)


# In[14]:


# Resultados de comparación de modelos
model_results = {
    "Modelo": ["Random Forest", "XGBoost"],
    "Accuracy": [0.90, 0.85],
    "Recall (Clase 1)": [0.96, 0.93],
    "Precision (Clase 1)": [0.84, 0.79]
}

df_models = pd.DataFrame(model_results)
print(df_models)


# Los experimentos permiten analizar dos aspectos principales:  
# 1. **Impacto de la imputación de valores faltantes**  
#    - El imputador **constant** alcanzó el mejor desempeño con ~90% de accuracy.  
#    - Las estrategias **median** y **most_frequent** también mostraron buenos resultados (~88%), mientras que **mean** y **KNN** estuvieron ligeramente por debajo (~87%).  
# 
# 2. **Comparación entre Random Forest y XGBoost**  
#    - **Random Forest** logró un accuracy mayor (0.90) en comparación con **XGBoost** (0.85).  
#    - En la clase positiva (pacientes con enfermedad), ambos modelos obtuvieron un **recall alto** (0.96 en RF y 0.93 en XGB), algo importante en las aplicaciones médicas para reducir falsos negativos.   
#    - De todas formas el Random Forest mantuvo un mejor equilibrio entre *precision* y *recall*, lo que lo hace más confiable en este caso.  

# ## Conclusiones

# 
# - El análisis del **Cleveland Heart Disease dataset** mostró que, aunque la presencia de valores faltantes fue mínima (únicamente en `ca` y `thal`), la estrategia de tratamiento tuvo un impacto medible en el desempeño. La imputación **constant** (reemplazo por 0) fue la que alcanzó el mejor resultado, con una accuracy cercana al 90%.  
# - En la comparación de los algoritmos implementados, Random Forest superó consistentemente a XGBoost, alcanzando no solo mayor accuracy, sino también un mejor balance entre *precision* (0.84) y *recall* (0.96). Este equilibrio es especialmente relevante en el ámbito clínico, donde es crucial minimizar los falsos negativos para no omitir pacientes con riesgo.  
# - **XGBoost**, aunque obtuvo un desempeño ligeramente inferior (~85% de accuracy), demostró una buena capacidad predictiva con un recall elevado (0.93), lo que confirma su solidez como una alternativa
# - Los resultados evidencian que el preprocesamiento de datos es un paso fundamental, tanto como la elección del modelo ya que pequeñas decisiones metodológicas pueden marcar diferencias significativas en el rendimiento final.  
# - En conclusión, la combinación de **Random Forest e imputación constant** fue la estrategia más efectiva para este conjunto de datos, resaltando la importancia de pipelines bien diseñados que integren limpieza de datos, transformación acertada y selección acertada de algoritmos para problemas de clasificación médica.
# 
