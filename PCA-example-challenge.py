#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# ### PCA in Machine Learning Workflows<br>
# #### Machine Learning I - Maestrí­a en Analítica Aplicada<br>
# #### Universidad de la Sabana<br>
# #### Prof: Hugo Franco<br>
# #### Example: Principal Component Analysis

# In[1]:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import time


# The IRIS dataset is used to illustrate the usage of PCA in a Supervised Learning pipeline

# In[5]:

# Load and prepare the iris dataset

# In[2]:


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names


# 1. Train baseline k-NN model

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Baseline model without PCA

# In[4]:


start_time = time.time()
baseline_model = KNeighborsClassifier(n_neighbors=3)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_time = time.time() - start_time


# In[5]:


print("Baseline Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
print(f"Training time: {baseline_time:.4f} seconds\n")


# In[6]:

# 2. Create and train Pipeline with PCA

# In[6]:


pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])


# In[7]:


start_time = time.time()
pca_pipeline.fit(X_train, y_train)
pipeline_pred = pca_pipeline.predict(X_test)
pipeline_time = time.time() - start_time


# In[8]:


print("PCA Pipeline Performance:")
print(f"Accuracy: {accuracy_score(y_test, pipeline_pred):.4f}")
print(f"Training time: {pipeline_time:.4f} seconds\n")


# In[7]:

# 3. Analyze explained variance ratio

# In[9]:


pca = PCA()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca.fit(X_scaled)


# Plot cumulative explained variance

# In[10]:


plt.figure(figsize=(10, 6))
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()


# In[8]:

# 4. Visualize 2D projection

# In[11]:


pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)


# In[12]:


plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
plt.xlabel(f'First Principal Component')
plt.ylabel(f'Second Principal Component')
plt.title('Iris Dataset - First Two Principal Components')
plt.colorbar(scatter)
plt.show()


# Print the explained variance ratio for the first two components

# In[13]:


print("Explained variance ratio for first two components:")
print(f"PC1: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"Total: {sum(pca_2d.explained_variance_ratio_):.4f}")


# #### Class activity - Workshop 3 Challenge: <br>
# 1. Add and organize this example according to the Data Science Workflow<br>
# 2. Apply the workflow to the wine dataset<br>
# 3. Complete the steps in the Supervised Learning Workflow for Data Science according to data preparation and per-model requirements and recommendations in this course, up-to-date<br>
# 4. **Compare the classification performance using the complete set of original features and using only two PCA-transformed features.**<br>
# 5. Modify the example to perform only a binary classification (good > 6) and compare your results with the multiclass performance

# ##### Wine Dataset Description<br>
# The Wine Quality dataset contains features like acidity, pH, alcohol content, and quality ratings. We'll convert the quality ratings into a binary classification problem.<br>
# <br>
# * Number of instances: 1599<br>
# * Features: 11 physicochemical properties<br>
# * Target: Binary (Good/Poor quality) and multiclass (Poor, Fair, and Good quality)<br>
# * Features include:<br>
# * Fixed acidity<br>
# * Volatile acidity<br>
# * Citric acid<br>
# * Residual sugar<br>
# * Chlorides<br>
# * Free sulfur dioxide<br>
# * Total sulfur dioxide<br>
# * Density<br>
# * pH<br>
# * Sulphates<br>
# * Alcohol

# In[11]:

# In[21]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns


# Load wine quality dataset

# In[28]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')


# Convert quality scores to three classes

# In[29]:


def quality_to_class(quality):
    if quality <= 5:
        return 'poor'
    elif quality <= 6:
        return 'fair'
    else:
        return 'good'


# Add new column with three classes

# In[31]:


df['quality_class'] = df['quality'].apply(quality_to_class)


# Show distribution of new classes

# In[32]:


print("Three-Class Distribution:\n", df['quality_class'].value_counts())


# Visualize class distribution

# In[33]:


plt.figure()
sns.countplot(data=df, x='quality_class', order=['poor', 'fair', 'good'])
plt.title('Wine Quality Class Distribution')
# plt.show()


# In[12]:

# In[34]:


print(df.head())
print(df.info())


# ## 2. Transformación de la variable objetivo<br>
# <br>
# El atributo `quality` se convertirá¡ en tres clases:<br>
# - Poor: calidad 5<br>
# - Fair: calidad = 6<br>
# - Good: calidad 7<br>
# <br>
# Esto convierte el problema en una **clasificaciÃ³n multiclase**.<br>
# 

# In[13]:

# Transformación en clases

# In[ ]:


def quality_to_class(quality):
    if quality <= 5:
        return 'poor'
    elif quality == 6:
        return 'fair'
    else:
        return 'good'


# In[ ]:


df['quality_class'] = df['quality'].apply(quality_to_class)


# VisualizaciÃ³n de clases

# In[ ]:


sns.countplot(data=df, x='quality_class', order=['poor','fair','good'])
plt.title("Distribución de clases de vino")
plt.show()


# ## 3. Separación en features y target<br>
# <br>
# Separamos las variables predictoras (X) de la variable objetivo categórica (y).<br>
# 

# In[14]:

# In[ ]:


X = df.drop(['quality','quality_class'], axis=1)
y = df['quality_class']


# Codificar clases

# In[ ]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)


# Train-Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# ## 4. Modelo base (sin PCA)<br>
# <br>
# Entrenamos un modelo **k-NN** usando todas las variables originales.<br>
# 

# In[15]:

# In[ ]:


start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_base = knn.predict(X_test)
base_time = time.time() - start_time


# In[ ]:


print("Modelo base (sin PCA)")
print("Accuracy:", accuracy_score(y_test, y_pred_base))
print("Tiempo de entrenamiento:", base_time)
print(classification_report(y_test, y_pred_base, target_names=le.classes_))


# ## 5. Modelo con PCA<br>
# <br>
# Se reduce la dimensionalidad a dos componentes principales y volvemos a entrenar el modelo k-NN.<br>
# 

# In[16]:

# In[ ]:


pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])


# In[ ]:


start_time = time.time()
pca_pipeline.fit(X_train, y_train)
y_pred_pca = pca_pipeline.predict(X_test)
pca_time = time.time() - start_time


# In[ ]:


print("Modelo con PCA (2 componentes)")
print("Accuracy:", accuracy_score(y_test, y_pred_pca))
print("Tiempo de entrenamiento:", pca_time)
print(classification_report(y_test, y_pred_pca, target_names=le.classes_))


# ## 6. Varianza explicada<br>
# <br>
# Se analiza la proporción de varianza retenida por los componentes principales.<br>
# 

# In[17]:

# In[ ]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


pca = PCA()
pca.fit(X_scaled)


# In[ ]:


plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Explicada Acumulada")
plt.title("Varianza explicada por PCA")
plt.grid(True)
plt.show()


# In[ ]:


print("Varianza explicada por los dos primeros componentes:", sum(pca.explained_variance_ratio_[:2]))


# ## 7. Visualización en 2D<br>
# <br>
# Se proyectan los datos en el espacio de los dos primeros componentes principales.<br>
# 

# In[18]:

# In[ ]:


X_pca_2d = PCA(n_components=2).fit_transform(X_scaled)


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca_2d[:,0], y=X_pca_2d[:,1], hue=y, palette="viridis")
plt.title("Proyección PCA en 2D")
plt.xlabel("Primer Componente Principal")
plt.ylabel("Segundo Componente Principal")
plt.show()


# ## 8. Resultados y Discusión

# Los experimentos realizados permiten analizar el impacto del **PCA** en la clasificación de la calidad del vino:  
# 
# ### Modelo base (sin PCA)  
# - Exactitud: **0.55**  
# - La clase *poor* obtuvo el mejor **recall** (0.66), mientras que la clase *good* presentó un desempeño bajo (**recall = 0.30**).  
# - Usando todas las variables originales, el modelo logra captar mejor los vinos de baja calidad, pero tiene dificultades para diferenciar las clases superiores.  
# 
# ### Modelo con PCA (2 componentes)  
# - Exactitud: **0.52** (ligeramente menor que el modelo base).  
# - La clase *good* redujo su desempeño (**recall = 0.28**), evidenciando pérdida de información relevante al usar solo dos componentes principales.  
# - El tiempo de entrenamiento fue menor, lo cual representa una ventaja computacional.  
# 
# ### Varianza explicada  
# - Los dos primeros componentes principales retienen aproximadamente **46% de la varianza total**.  
# - Esto implica que más de la mitad de la información se pierde en la proyección 2D, lo que explica la caída en el rendimiento.  
# 
# ### Visualización en 2D  
# - El gráfico muestra cierto agrupamiento entre clases (*poor*, *fair*, *good*), pero con **alta superposición**.  
# - Se confirma que **dos componentes no son suficientes** para separar las clases de manera clara en este dataset.  
# 
# 
# 
# 
# ## 8.1 Clasificación Binaria (good > 6)  
# 
# Para simplificar el problema, se transformó la variable objetivo en binaria:  
# - **0 = Poor/Fair (calidad ≤ 6)**  
# - **1 = Good (calidad > 6)**  
# 
# ### Modelo base binario (sin PCA)  
# - Se obtuvo una exactitud mayor que en la clasificación multiclase.  
# - La separación entre vinos buenos y no buenos fue más clara, especialmente en la clase *Good*.  
# 
# ### Modelo con PCA binario (2 componentes)  
# - El desempeño fue similar, aunque se observó una ligera reducción de métricas debido a la pérdida de información.  
# - Sin embargo, el tiempo de entrenamiento se redujo, manteniendo la ventaja computacional observada previamente.  
# 
# **Conclusión parcial:**  
# Al pasar de un problema multiclase a uno binario, los modelos mejoran su desempeño en la detección de vinos de calidad aceptable. Sin embargo, la reducción de dimensionalidad con PCA sigue implicando pérdida de información relevante, lo cual impacta el rendimiento predictivo.  
# 

# In[35]:


# Transformamos la variable objetivo en binaria
df['binary_quality'] = df['quality'].apply(lambda x: 1 if x > 6 else 0)

# Features y target
X_bin = df.drop(['quality', 'quality_class', 'binary_quality'], axis=1)
y_bin = df['binary_quality']

# Split
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# Modelo base binario (sin PCA)
knn_bin = KNeighborsClassifier(n_neighbors=5)
knn_bin.fit(X_train_bin, y_train_bin)
y_pred_bin = knn_bin.predict(X_test_bin)

print("Clasificación binaria (sin PCA)")
print("Accuracy:", accuracy_score(y_test_bin, y_pred_bin))
print(classification_report(y_test_bin, y_pred_bin, target_names=["Poor/Fair", "Good"]))

# Modelo con PCA (binario)
pca_bin = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pca_bin.fit(X_train_bin, y_train_bin)
y_pred_bin_pca = pca_bin.predict(X_test_bin)

print("\nClasificación binaria (con PCA, 2 componentes)")
print("Accuracy:", accuracy_score(y_test_bin, y_pred_bin_pca))
print(classification_report(y_test_bin, y_pred_bin_pca, target_names=["Poor/Fair", "Good"]))


# ### Resultados de la Clasificación Binaria
# 
# Los experimentos con la variable objetivo transformada en binaria (0 = *Poor/Fair*, 1 = *Good*) permiten observar lo siguiente:
# 
# **Modelo base binario (sin PCA):**
# - Exactitud: **0.86**
# - La clase *Poor/Fair* tuvo un desempeño sobresaliente (**recall = 0.96**), mostrando que el modelo identifica con mucha seguridad los vinos de baja calidad.
# - La clase *Good* obtuvo un desempeño más limitado (**recall = 0.23**), lo cual indica dificultad en detectar vinos de buena calidad.
# 
# **Modelo con PCA binario (2 componentes):**
# - Exactitud: **0.86** (similar al modelo sin PCA).
# - La clase *Good* mostró una leve mejora (**recall = 0.28** frente a 0.23).
# - Se mantuvo la ventaja computacional de PCA, aunque con una ligera reducción de información.

# ## 9. Conclusiones
# 
# - El uso de PCA permitió reducir la dimensionalidad y facilitar la visualización en 2D, pero a costa de perder información relevante. Con solo dos componentes principales se retuvo ~46% de la varianza, lo cual explica la ligera disminución en el rendimiento predictivo.  
# - En el escenario multiclase, el modelo base sin PCA logró un mejor balance entre *precisión* y *recall*, aunque con dificultades para diferenciar las clases de mayor calidad (*Good*).  
# - En la clasificación binaria (Good > 6), los modelos alcanzaron una exactitud cercana al 86%, mostrando un mejor desempeño global respecto al escenario multiclase. Sin embargo, la clase positiva (*Good*) siguió siendo más difícil de identificar con métricas bajas de recall y precisión.  
# - Comparando modelos con y sin PCA en el caso binario, se observó que PCA mantiene la exactitud y aporta eficiencia computacional, aunque no soluciona del todo las limitaciones de clasificación en la clase *Good*.  
# - En conjunto, el taller evidencia la importancia de:  
#   1. Evaluar métricas más allá de la exactitud (recall, precision, f1-score).  
#   2. Considerar el impacto de técnicas de reducción de dimensionalidad como PCA.  
#   3. Analizar la naturaleza del problema (multiclase vs binario) para escoger la representación y el modelo más adecuados.
# 
#  
# El PCA es útil para visualización y eficiencia, pero el modelo base conserva mejor poder predictivo. La simplificación a clasificación binaria mejora la exactitud global, aunque sigue existiendo el reto de identificar con mayor precisión los vinos de calidad superior.
# 
