import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Clasificación de frutas con k-NN
def classify_fruit():
    data = np.array([
        [150, 7, 1],  # Manzana
        [170, 8, 1],  # Manzana
        [130, 6, 1],  # Naranja
        [180, 8.5, 0] # Naranja
    ])

    X = data[:, :2]  # Características (Peso, Diámetro)
    y = data[:, 2]   # Etiquetas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    new_fruit = np.array([[160, 7.5]])  # Nueva fruta
    prediction = knn.predict(new_fruit)
    fruit_type = "Manzana" if prediction[0] == 1 else "Naranja"

    print(f"Resultado k-NN: La nueva fruta probablemente es una: {fruit_type}")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label="Datos conocidos")
    plt.scatter(new_fruit[0, 0], new_fruit[0, 1], color='red', label="Fruta nueva", s=100)
    plt.xlabel("Peso (g)")
    plt.ylabel("Diámetro (cm)")
    plt.legend()
    plt.title("Clasificación de Frutas con k-NN")
    plt.show()

if __name__ == "__main__":
    classify_fruit()
