import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Definición del problema y parámetros
POBLACION_SIZE = 50
CROMOSOMA_LEN = 16 
GENERACIONES = 100
PROB_CRUCE = 0.8
PROB_MUTACION = 0.05

# 2. Función de Aptitud
def aptitud(x):
    return x * np.sin(10 * np.pi * x) + 1

# Decodificación de binario a real en el rango [0, 1]
def decodificar(cromosoma):
    valor_decimal = int("".join(str(i) for i in cromosoma), 2)
    max_decimal = (2**CROMOSOMA_LEN) - 1
    return valor_decimal / max_decimal

# 3. Operadores Genéticos
def crear_poblacion(size, length):
    return [[random.randint(0, 1) for _ in range(length)] for _ in range(size)]

def seleccion_ruleta(poblacion, aptitudes):
    suma_aptitud = sum(aptitudes)
    probabilidades = [apt / suma_aptitud for apt in aptitudes]
    seleccionados = random.choices(poblacion, weights=probabilidades, k=len(poblacion))
    return seleccionados

def cruce(padre1, padre2):
    if random.random() < PROB_CRUCE:
        punto = random.randint(1, CROMOSOMA_LEN - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1, padre2

def mutacion(cromosoma):
    for i in range(CROMOSOMA_LEN):
        if random.random() < PROB_MUTACION:
            cromosoma[i] = 1 - cromosoma[i]
    return cromosoma

# 4. Ciclo Principal del AG
poblacion = crear_poblacion(POBLACION_SIZE, CROMOSOMA_LEN)
mejores_aptitudes = []

for gen in range(GENERACIONES):
    valores_x = [decodificar(ind) for ind in poblacion]
    aptitudes = [aptitud(x) for x in valores_x]
    
    mejor_aptitud_gen = max(aptitudes)
    mejores_aptitudes.append(mejor_aptitud_gen)
    
    poblacion_seleccionada = seleccion_ruleta(poblacion, aptitudes)
    
    nueva_poblacion = []
    for i in range(0, POBLACION_SIZE, 2):
        hijo1, hijo2 = cruce(poblacion_seleccionada[i], poblacion_seleccionada[i+1])
        nueva_poblacion.extend([mutacion(hijo1), mutacion(hijo2)])
        
    poblacion = nueva_poblacion

# Resultados
mejor_individuo = max(poblacion, key=lambda ind: aptitud(decodificar(ind)))
mejor_x = decodificar(mejor_individuo)
print(f"Mejor x encontrado: {mejor_x:.4f}")
print(f"Valor máximo f(x): {aptitud(mejor_x):.4f}")

plt.plot(mejores_aptitudes)
plt.title("Evolución de la Aptitud - Ejercicio 1")
plt.xlabel("Generaciones")
plt.ylabel("Mejor Aptitud")
plt.grid(True)
plt.show()