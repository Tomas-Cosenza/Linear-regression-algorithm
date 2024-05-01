
# =============================================================================
# Generación del conjunto de datos
# =============================================================================


import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

print("La longitud del conjunto de datos es:", len(X))

# =============================================================================
#  Visualizacion del conjunto de datos
# =============================================================================


import matplotlib.pyplot as plt


plt.plot(X, y, "b.")
plt.show()

plt.plot(X, y, "b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.show()

# =============================================================================
# Modificacion del conjunto de datos
# =============================================================================

import pandas as pd

data = {'n_equipos_afectados': X.flatten(), 'coste': y.flatten()}
df = pd.DataFrame(data)
df.head(10)

# Escalado del numero de equipos afectados
df['n_equipos_afectados'] = df['n_equipos_afectados'] * 1000
df['n_equipos_afectados'] = df['n_equipos_afectados'].astype('int')
# Escalado del coste
df['coste'] = df['coste'] * 10000
df['coste'] = df['coste'].astype('int')
df.head(10)

# Representacion grafica del conjunto de datos
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()

# =============================================================================
# Construccion del modelo
# =============================================================================

import sklearn

from sklearn.linear_model import LinearRegression

# Construccion del modelo y ajuste de la funcion hipotesis
lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)

# Parametro theta 0
lin_reg.intercept_

# Parametro theta 1
lin_reg.coef_

# Prediccion para el valor mi­nimo y maximo del conjunto de datos de entrenamiento
X_min_max = np.array([[df["n_equipos_afectados"].min()], [df["n_equipos_afectados"].max()]])
y_train_pred = lin_reg.predict(X_min_max)

# Representacion grafica de la funcion hipotesis generada
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()

# =============================================================================
# Prediccion de nuevos ejemplos
# =============================================================================

x_new = np.array([[1300]]) # 1300 equipos afectados

# Prediccion del coste que tendrí­a el incidente
coste = lin_reg.predict(x_new)

print("El coste del incidente serÃ­a:", int(coste[0]), "â‚¬")

plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(x_new, coste, "rx")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()

