import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Parámetros del escenario
NUM_PARTIDOS = 5
NUM_CURULES = 50
NUM_ENTIDADES = 50

curules_partidos = np.random.multinomial(NUM_CURULES, np.ones(NUM_PARTIDOS)/NUM_PARTIDOS)
porcentaje_poder_objetivo = curules_partidos / NUM_CURULES
pesos_entidades = np.random.randint(1, 101, NUM_ENTIDADES)
poder_total_entidades = sum(pesos_entidades)

# 2. Representación
POB_SIZE = 100
GENS = 200

# 3. Función de Aptitud
def calcular_aptitud(cromosoma):
    poder_asignado = np.zeros(NUM_PARTIDOS)
    for i, partido in enumerate(cromosoma):
        poder_asignado[partido] += pesos_entidades[i]
    
    porcentaje_asignado = poder_asignado / poder_total_entidades
    error = np.sum((porcentaje_asignado - porcentaje_poder_objetivo)**2)
    return 1 / (1 + error)

# 4. AG Principal
poblacion = [np.random.randint(0, NUM_PARTIDOS, NUM_ENTIDADES).tolist() for _ in range(POB_SIZE)]
historial_aptitud = []

for gen in range(GENS):
    aptitudes = [calcular_aptitud(ind) for ind in poblacion]
    historial_aptitud.append(max(aptitudes))
    
    nueva_poblacion = []
    for _ in range(POB_SIZE):
        torneo = random.sample(list(zip(poblacion, aptitudes)), 3)
        ganador = max(torneo, key=lambda x: x[1])[0]
        nueva_poblacion.append(ganador.copy())
    
    for i in range(0, POB_SIZE, 2):
        if random.random() < 0.7:
            punto = random.randint(1, NUM_ENTIDADES-1)
            temp = nueva_poblacion[i][punto:].copy()
            nueva_poblacion[i][punto:] = nueva_poblacion[i+1][punto:]
            nueva_poblacion[i+1][punto:] = temp
            
    for i in range(POB_SIZE):
        for j in range(NUM_ENTIDADES):
            if random.random() < 0.02:
                nueva_poblacion[i][j] = random.randint(0, NUM_PARTIDOS-1)
                
    poblacion = nueva_poblacion

mejor_asignacion = max(poblacion, key=calcular_aptitud)
print("Distribución Objetivo de Curules:", curules_partidos)
poder_final = np.zeros(NUM_PARTIDOS)
for i, partido in enumerate(mejor_asignacion):
    poder_final[partido] += pesos_entidades[i]
print("Porcentaje de Poder Objetivo:   ", np.round(porcentaje_poder_objetivo*100, 2))
print("Porcentaje de Poder Alcanzado:  ", np.round((poder_final/poder_total_entidades)*100, 2))

plt.plot(historial_aptitud)
plt.title("Evolución de Asignación de Poder - Ejercicio 2")
plt.show()