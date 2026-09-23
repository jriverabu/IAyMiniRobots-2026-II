import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Definición del problema
capacidades = np.array([3, 6, 5, 4]) 
demandas = np.array([4, 3, 5, 3]) 
costos_transporte = np.array([
    [1, 4, 3, 6],
    [4, 1, 4, 5],
    [3, 4, 1, 4],
    [6, 5, 4, 1]
])
costos_gen = np.array([680, 720, 660, 750])

# 2. Representación
POB_SIZE = 150
GENS = 300

def crear_individuo():
    ind = np.zeros((4, 4), dtype=int)
    for j in range(4): 
        restante = demandas[j]
        while restante > 0:
            i = random.randint(0, 3) 
            # Corrección aplicada: Evitar bucle infinito cambiando el límite inferior a 1
            aporte = random.randint(1, restante) 
            ind[i][j] += aporte
            restante -= aporte
    return ind.flatten().tolist()

# 3. Función de Aptitud
def aptitud_energia(cromosoma):
    matriz = np.array(cromosoma).reshape(4, 4)
    costo_total = 0
    penalizacion = 0
    
    gen_por_planta = np.sum(matriz, axis=1)
    recibido_por_ciudad = np.sum(matriz, axis=0)
    
    for i in range(4):
        if gen_por_planta[i] > capacidades[i]:
            penalizacion += (gen_por_planta[i] - capacidades[i]) * 5000
            
    for j in range(4):
        if recibido_por_ciudad[j] != demandas[j]:
            penalizacion += abs(recibido_por_ciudad[j] - demandas[j]) * 10000
            
    for i in range(4):
        for j in range(4):
            costo_total += matriz[i][j] * (costos_transporte[i][j] + costos_gen[i])
            
    return 1.0 / (costo_total + penalizacion + 1)

# 4. Ejecución del Algoritmo
poblacion = [crear_individuo() for _ in range(POB_SIZE)]
historial_costos = []

for gen in range(GENS):
    aptitudes = [aptitud_energia(ind) for ind in poblacion]
    mejor_ind = poblacion[np.argmax(aptitudes)]
    
    mejor_matriz = np.array(mejor_ind).reshape(4, 4)
    costo_real = sum(mejor_matriz[i][j] * (costos_transporte[i][j] + costos_gen[i]) 
                     for i in range(4) for j in range(4))
    historial_costos.append(costo_real)
    
    nueva_poblacion = [mejor_ind.copy()] 
    for _ in range(POB_SIZE - 1):
        torneo = random.sample(list(zip(poblacion, aptitudes)), 4)
        ganador = max(torneo, key=lambda x: x[1])[0]
        nueva_poblacion.append(ganador.copy())
        
    # Cruce 
    for i in range(1, POB_SIZE, 2):
        if i+1 < POB_SIZE and random.random() < 0.8:
            punto = random.randint(1, 14)
            temp = nueva_poblacion[i][punto:].copy()
            nueva_poblacion[i][punto:] = nueva_poblacion[i+1][punto:]
            nueva_poblacion[i+1][punto:] = temp
            
    # Corrección aplicada: Mutación desindentada y aplicada correctamente
    for i in range(1, POB_SIZE):
        for j in range(16):
            if random.random() < 0.05:
                nueva_poblacion[i][j] = max(0, nueva_poblacion[i][j] + random.choice([-1, 1]))
                
    poblacion = nueva_poblacion

mejor_solucion = np.array(max(poblacion, key=aptitud_energia)).reshape(4, 4)
print("Mejor despacho de energía encontrado (Filas=Plantas, Cols=Ciudades):\n", mejor_solucion)
print("Costos totales (Mínimos alcanzados): $", historial_costos[-1])

plt.plot(historial_costos)
plt.title("Minimización de Costos de Transporte y Generación")
plt.ylabel("Costo Total")
plt.xlabel("Generaciones")
plt.show()