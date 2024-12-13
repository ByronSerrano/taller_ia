import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Crear un conjunto de datos de ejemplo
data = {
    'Distancia_Recorrida': [50, 100, 150, 200, 250],
    'Peso_Vehiculo': [1000, 1500, 1800, 2000, 2200],
    'Velocidad_Promedio': [60, 80, 90, 100, 110],
    'Consumo_Gasolina': [5, 7, 9, 11, 13]
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Características (X) y objetivo (y)
X = df[['Distancia_Recorrida', 'Peso_Vehiculo', 'Velocidad_Promedio']].values  # Usamos .values para convertir a numpy array
y = df['Consumo_Gasolina'].values  # Usamos .values para convertir a numpy array

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Predicción para un coche con distancia de 150 km, peso 1800 kg y velocidad 90 km/h
nuevo_coche = np.array([[150, 1800, 90]])  # Usamos un arreglo numpy
prediccion_consumo = model.predict(nuevo_coche)

print(f"El consumo estimado de gasolina para este coche es: {prediccion_consumo[0]:.2f} litros")

# Visualizar la relación entre distancia y consumo (para ilustrar un ejemplo)
plt.scatter(df['Distancia_Recorrida'], y, color='blue', label='Datos')
plt.plot(df['Distancia_Recorrida'], model.predict(X), color='red', label='Regresión Lineal')
plt.xlabel('Distancia Recorrida (km)')
plt.ylabel('Consumo de Gasolina (litros)')
plt.title('Regresión Lineal: Consumo de Gasolina vs Distancia Recorrida')
plt.legend()
plt.show()