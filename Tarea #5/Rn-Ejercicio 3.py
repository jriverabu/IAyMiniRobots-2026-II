import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# 1. DESCARGA / CARGA DEL DATASET (Dataset oficial de Kaggle)
cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = cancer_data.target

print(f"Dimensiones de la matriz de características X: {X.shape}")
print(f"Distribución de etiquetas (0 = Maligno, 1 = Benigno):\n{pd.Series(y).value_counts()}")

# 2. DIVISIÓN DE DATOS Y NORMALIZACIÓN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalamiento estándar (Z-score normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. CREACIÓN DEL MODELO DE CLASIFICACIÓN
model_cancer = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),  # Regularización para evitar Overfitting
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Salida binaria (0 a 1)
])

model_cancer.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)

# 4. ENTRENAMIENTO DEL MODELO
history = model_cancer.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.15,
    verbose=0
)

# 5. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
test_loss, test_acc, test_recall = model_cancer.evaluate(X_test_scaled, y_test, verbose=0)

y_pred_probs = model_cancer.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\n=== REPORTE DE CLASIFICACIÓN EN DATOS DE PRUEBA ===")
print(classification_report(y_test, y_pred, target_names=cancer_data.target_names))

print("=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

# 6. ANÁLISIS DE UN EJEMPLO NUEVO (PACIENTE DE PRUEBA)
paciente_ejemplo = X_test_scaled[0].reshape(1, -1)
pred_prob = model_cancer.predict(paciente_ejemplo)[0][0]
diag_real = "Benigno" if y_test[0] == 1 else "Maligno"
diag_pred = "Benigno" if pred_prob > 0.5 else "Maligno"

print(f"\n--- Análisis de Diagnóstico para Paciente de Muestra ---")
print(f"Probabilidad asignada por la red (Benignidad): {pred_prob * 100:.2f}%")
print(f"Diagnóstico Predicho: {diag_pred}")
print(f"Diagnóstico Real: {diag_real}")