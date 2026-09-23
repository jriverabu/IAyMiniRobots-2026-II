import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. CARGA DEL DATASET
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Nombres de las clases según la documentación oficial
class_names = ['Camiseta/Top', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

# 2. PREPROCESAMIENTO
# Escalar los píxeles de [0, 255] a un rango de [0.0, 1.0]
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. CONSTRUCCIÓN DE LA ARQUITECTURA DE LA RED
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Capa de entrada: aplana 28x28 a 784 píxeles
    keras.layers.Dense(128, activation='relu'),   # Capa oculta 1: 128 neuronas ReLU
    keras.layers.Dense(64, activation='relu'),    # Capa oculta 2: 64 neuronas ReLU
    keras.layers.Dense(10, activation='softmax')  # Capa de salida: 10 probabilidades
])

# 4. COMPILACIÓN DEL MODELO
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. ENTRENAMIENTO
print("--- Iniciando entrenamiento ---")
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# 6. EVALUACIÓN CON EL CONJUNTO DE PRUEBA
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nPrecisión en el conjunto de prueba (Test Accuracy): {test_acc * 100:.2f}%")

# 7. MATRIZ Y EJEMPLO DE PREDICCIÓN
predictions = model.predict(test_images)
ejemplo_idx = 0
print(f"\nPredicción para la primera imagen de prueba:")
print(f"Clase Real: {class_names[test_labels[ejemplo_idx]]}")
print(f"Clase Predicha: {class_names[np.argmax(predictions[ejemplo_idx])]} "
      f"(Probabilidad: {np.max(predictions[ejemplo_idx])*100:.2f}%)")