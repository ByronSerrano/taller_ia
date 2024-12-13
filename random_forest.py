import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def play_tennis_decision():
    data = pd.DataFrame({
        "Clima": ["Soleado", "Nublado", "Lluvioso", "Soleado"],
        "Temperatura": ["Alta", "Alta", "Baja", "Baja"],
        "Humedad": ["Alta", "Alta", "Baja", "Alta"],
        "JugarTenis": [0, 1, 1, 0]
    })

    data_encoded = pd.get_dummies(data.drop(columns=["JugarTenis"]))
    y = data["JugarTenis"]

    X_train, X_test, y_train, y_test = train_test_split(data_encoded, y, test_size=0.25, random_state=42)

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plot_tree(tree_clf, feature_names=data_encoded.columns, class_names=["No", "Sí"], filled=True)
    plt.title("Árbol de Decisión - Jugar Tenis")
    plt.show()

    y_pred = tree_clf.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred)
    print(f"Resultado Árbol de Decisión: Precisión del modelo: {accuracy_tree * 100:.2f}%")

    rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred_rf = rf_clf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Resultado Random Forest: Precisión del modelo: {accuracy_rf * 100:.2f}%")
