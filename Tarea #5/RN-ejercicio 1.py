import numpy as np

# ------------------------------------------------------------------------------
# CLASE BASE PARA RED NEURONAL DE 2 CAPAS OCULTAS (Clase_NNA3 modificada)
# ------------------------------------------------------------------------------
class RedNeuronal2CapasOcultas:
    def __init__(self, lr=0.5):
        self.lr = lr  # Tasa de aprendizaje (eta)
        
        # Inicialización de pesos y sesgos con valores aleatorios (semilla fija)
        np.random.seed(42)
        self.W1 = np.random.uniform(-1, 1, (3, 2))  # Capa Oculta 1: 3 neuronas, 2 entradas
        self.b1 = np.random.uniform(-1, 1, (3, 1))
        
        self.W2 = np.random.uniform(-1, 1, (2, 3))  # Capa Oculta 2: 2 neuronas, 3 entradas
        self.b2 = np.random.uniform(-1, 1, (2, 1))
        
        self.W3 = np.random.uniform(-1, 1, (1, 2))  # Capa Salida: 1 neurona, 2 entradas
        self.b3 = np.random.uniform(-1, 1, (1, 1))

    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoide(self, a):
        """Derivada de la sigmoide en función de la salida 'a' = σ(z)"""
        return a * (1 - a)

    def forward(self, X):
        """Propagación hacia adelante"""
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoide(self.Z1)
        
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoide(self.Z2)
        
        self.Z3 = np.dot(self.W3, self.A2) + self.b3
        self.A3 = self.sigmoide(self.Z3)
        
        return self.A3

    def backpropagation(self, X, Y):
        """Retropropagación del error aplicando regla de la cadena"""
        # Error en capa de salida
        error3 = Y - self.A3
        delta3 = error3 * self.d_sigmoide(self.A3)  # (1, 1)
        
        # Propagación a Capa Oculta 2
        error2 = np.dot(self.W3.T, delta3)
        delta2 = error2 * self.d_sigmoide(self.A2)  # (2, 1)
        
        # Propagación a Capa Oculta 1
        error1 = np.dot(self.W2.T, delta2)
        delta1 = error1 * self.d_sigmoide(self.A1)  # (3, 1)
        
        # Actualización de Pesos y Sesgos mediante Descenso del Gradiente
        self.W3 += self.lr * np.dot(delta3, self.A2.T)
        self.b3 += self.lr * delta3
        
        self.W2 += self.lr * np.dot(delta2, self.A1.T)
        self.b2 += self.lr * delta2
        
        self.W1 += self.lr * np.dot(delta1, X.T)
        self.b1 += self.lr * delta1

    def entrenar(self, X_data, Y_data, epocas=20000):
        for _ in range(epocas):
            for i in range(len(X_data)):
                x = X_data[i].reshape(-1, 1)
                y = Y_data[i].reshape(-1, 1)
                self.forward(x)
                self.backpropagation(x, y)

    def probar(self, X_data):
        print("Entrada (X1, X2) | Salida Predicha | Salida Redondeada")
        print("-" * 50)
        for x in X_data:
            x_vec = x.reshape(-1, 1)
            pred = self.forward(x_vec)[0, 0]
            print(f"    {x[0]} , {x[1]}       |    {pred:.6f}    |         {int(np.round(pred))}")

# ------------------------------------------------------------------------------
# DATOS DE LAS COMPUERTAS LÓGICAS (Entradas y Salidas Esperadas)
# ------------------------------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Tabla de Verdad NAND: Falsa solo si ambas entradas son 1
Y_NAND = np.array([[1], [1], [1], [0]])

# Tabla de Verdad XOR: Verdadera solo si una de las entradas es 1 (No separable linealmente)
Y_XOR = np.array([[0], [1], [1], [0]])

# ------------------------------------------------------------------------------
# EJECUCIÓN Y PRUEBA
# ------------------------------------------------------------------------------
print("=== RED NEURONAL PARA COMPUERTA NAND ===")
red_nand = RedNeuronal2CapasOcultas(lr=0.5)
red_nand.entrenar(X, Y_NAND, epocas=15000)
red_nand.probar(X)

print("\n=== RED NEURONAL PARA COMPUERTA XOR ===")
red_xor = RedNeuronal2CapasOcultas(lr=0.5)
red_xor.entrenar(X, Y_XOR, epocas=30000)
red_xor.probar(X)