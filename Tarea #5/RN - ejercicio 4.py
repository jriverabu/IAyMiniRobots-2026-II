#Diagnóstico y Clasificación Multiclase de Estados Operativos en Sensores Industriales: 
#Comparativa entre Implementación Desde Cero (NumPy) y Framework Comercial (Keras/TensorFlow).
#Desarrollar un sistema de clasificación multiclase que procese $4$ variables físicas de
#un equipo rotativo (vibración, temperatura, presión y velocidad de rotación) para diagnosticar $3$ 
#estados de operación: 0: Normal, 1: Desgaste Moderado, y 2: Falla Crítica. Se debe
#implementar la red neuronal en NumPy desde cero (incorporando la función de activación
#Softmax y la pérdida de Entropía Cruzada) y compararla en precisión, velocidad de 
#convergencia y arquitectura contra un modelo equivalente en Keras/TensorFlow.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# 1. GENERACIÓN DEL DATASET SINTÉTICO DE SENSORES
# ==============================================================================
# Generamos 1,200 muestras simulando 4 sensores continuos y 3 clases de fallas
X_raw, y_raw = make_classification(
    n_samples=1200,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# Nombres simbólicos de las características y estados
feature_names = ['Vibracion_mm_s', 'Temperatura_C', 'Presion_PSI', 'Velocidad_RPM']
class_names = ['Estado Normal', 'Desgaste Leve', 'Falla Critica']

# División en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# Normalización Z-score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversión de etiquetas a codificación One-Hot para NumPy
def to_one_hot(y, num_classes=3):
    return np.eye(num_classes)[y]

Y_train_oh = to_one_hot(y_train, 3)
Y_test_oh = to_one_hot(y_test, 3)


# ==============================================================================
# 2. PARTE A: RED NEURONAL MULTICLASE EN NUMPY (DESDE CERO)
# ==============================================================================
class RedMulticlaseNumPy:
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=3, lr=0.05):
        self.lr = lr
        np.random.seed(42)
        # Inicialización He/Xavier para evitar saturación de gradiente
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def d_relu(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        # Softmax numéricamente estable restando el máximo
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backpropagation(self, X, Y_onehot):
        m = X.shape[1]
        
        # Derivada combinada de Categorical Cross-Entropy con Softmax: dZ2 = A2 - Y
        dZ2 = self.A2 - Y_onehot
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.d_relu(self.Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Actualización de parámetros por Descenso del Gradiente
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def entrenar(self, X_data, Y_data, epocas=800):
        X_in = X_data.T  # Formato (características, muestras)
        Y_in = Y_data.T  # Formato (clases, muestras)
        for _ in range(epocas):
            self.forward(X_in)
            self.backpropagation(X_in, Y_in)

    def predecir(self, X_data):
        A2 = self.forward(X_data.T)
        return np.argmax(A2, axis=0)


# Entrenar modelo en NumPy
print("--- Entrenando Red Neuronal Multicase en NumPy (Desde Cero) ---")
red_numpy = RedMulticlaseNumPy(input_dim=4, hidden_dim=8, output_dim=3, lr=0.1)
red_numpy.entrenar(X_train_scaled, Y_train_oh, epocas=1000)
preds_numpy = red_numpy.predecir(X_test_scaled)

acc_numpy = np.mean(preds_numpy == y_test)
print(f"Precisión (Accuracy) de la red en NumPy: {acc_numpy * 100:.2f}%\n")


# ==============================================================================
# 3. PARTE B: RED NEURONAL EQUIVALENTE EN KERAS / TENSORFLOW
# ==============================================================================
print("--- Entrenando Red Neuronal Equivalente en Keras / TensorFlow ---")
model_tf = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])

model_tf.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_tf.fit(X_train_scaled, y_train, epochs=1000, batch_size=len(X_train_scaled), verbose=0)

loss_tf, acc_tf = model_tf.evaluate(X_test_scaled, y_test, verbose=0)
preds_tf = np.argmax(model_tf.predict(X_test_scaled, verbose=0), axis=1)

print(f"Precisión (Accuracy) de la red en Keras/TensorFlow: {acc_tf * 100:.2f}%\n")


# ==============================================================================
# 4. COMPARATIVA Y ANÁLISIS DE RESULTADOS
# ==============================================================================
print("=========================================================")
print("             REPORTE COMPARATIVO DE DESEMPEÑO           ")
print("=========================================================")
print("Matriz de Confusión - Red en NumPy:")
print(confusion_matrix(y_test, preds_numpy))

print("\nMatriz de Confusión - Red en Keras:")
print(confusion_matrix(y_test, preds_tf))

print("\nReporte de Clasificación en Keras (Detalle por Estado):")
print(classification_report(y_test, preds_tf, target_names=class_names))