"""
Librerías: DEAP (PG), numpy, pandas, scikit-learn (partición, métricas y
modelos de referencia), matplotlib (gráficas).
    pip install deap numpy pandas scikit-learn matplotlib

Ejecución
---------
    python ej3_pg_deteccion_fraudes.py
    python ej3_pg_deteccion_fraudes.py --n 30000 --tasa-fraude 0.015 --generaciones 80
    python ej3_pg_deteccion_fraudes.py --aptitud error     # aptitud f_apt del documento

Estructura del programa (= estructura de la solución)
-----------------------------------------------------
 PARTE A. Archivo aleatorio "ingenuo": cada columna se sortea de forma
          uniforme e independiente, y fraude = sí/no al 50 %.
 PARTE B. Revisión crítica: se miden y enumeran los defectos del archivo
          ingenuo (tasa de fraude irreal, distribuciones planas, ausencia de
          relación entre variables y etiqueta, montos incoherentes con el
          ingreso...). Se entrena PG sobre él para demostrar que no hay nada
          que aprender (AUC ~ 0.5).
 PARTE C. Archivo "realista": clientes con ingresos log-normales (piso en el
          salario mínimo), montos proporcionales al ingreso, perfil horario
          diurno, distancias mayoritariamente cortas, tasa de fraude baja
          (2 % por defecto) y tres PATRONES de fraude (compra no presencial de
          alto valor, "prueba" de tarjeta y fraude camuflado), más ruido de
          etiqueta. Se comparan ambas versiones en tabla y gráfica.
 PARTE D. Modelo por PG (los seis pasos preparatorios del documento):
    1. Problema: clasificación binaria desbalanceada (fraude / legítima).
    2. Terminales T = { M, I, D, HC, HS, R }
          M  = log10(monto)      estandarizado
          I  = log10(ingreso)    estandarizado
          D  = log10(1+dist_km)  estandarizado
          HC = cos(2*pi*hora/24), HS = sin(2*pi*hora/24)   (hora cíclica:
               las 23 h y las 0 h quedan cerca, lo que no pasa con 'hora')
          R  = constantes aleatorias efímeras en [-2, 2]
       (los logaritmos domestican las colas largas de montos e ingresos).
    3. Funciones F = { +, -, *, /protegida, neg, max, min, si_mayor(a,b,c,d) }
       si_mayor(a,b,c,d) = c si a > b, si no d   (como "If a<b?c:d" de MEPX).
       Clausura: todas reciben y devuelven reales; la división protegida
       devuelve 1 si el denominador es ~0.
    4. Aptitud: el programa-árbol produce un PUNTAJE de riesgo s(x).
       - 'auc' (por defecto): área bajo la curva ROC del puntaje en
         entrenamiento, menos una penalización de parsimonia por nodo.
         Se prefiere a la exactitud porque con 2 % de fraudes el modelo
         trivial "nunca hay fraude" tendría 98 % de exactitud y 0 detección.
       - 'error': la f_apt del documento, f_apt = 1/(0.1 + E), con
         E = error absoluto medio PONDERADO por clase entre y_j y la
         probabilidad sigmoide(s_j) (el peso equilibra fraudes y legítimas).
    5. Parámetros: población 400, 60 generaciones, pc = 0.7, pm = 0.3,
       doble torneo (aptitud 3 / parsimonia 1.2), elitismo 3, altura <= 8,
       penalización de parsimonia λ = 0.0002 por nodo, 3 corridas
       independientes (se conserva la de mejor AUC en validación).
    6. Resultado: de los mejores de cada generación se designa el de mayor
       AUC en VALIDACIÓN (evita sobreajuste). El umbral de decisión se fija
       en validación maximizando F1. Se reporta en el conjunto de PRUEBA.
 PARTE E. Evaluación: matriz de confusión, precisión, sensibilidad (recall),
          F1, AUC-ROC, AUC-PR, costo monetario; comparación con regresión
          logística y árbol de decisión; importancia por permutación.
 PARTE F. Exportación del modelo como módulo Python autónomo
          (modelo_fraude_pg.py), igual que MEPX genera 'cancer.py'.

Salidas en ./resultados_ej3/
    transacciones_ingenuo.csv, transacciones_realista.csv, reporte_ej3.txt,
    distribuciones_ingenuo_vs_realista.png, convergencia_pg_fraude.png,
    curvas_roc_pr.png, modelo_fraude_pg.py
===============================================================================
"""

import argparse
import math
import operator
import os
import random
import sys
import time
from functools import partial

try:
    from deap import base, creator, gp, tools
except ImportError:
    sys.exit("Falta la librería DEAP. Instálela con:  pip install deap")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Salario mínimo de referencia (SMMLV 2025 = 1.423.500 COP). Ajustar al valor vigente.
SALARIO_MINIMO = 1_423_500
COSTO_REVISION = 15_000          # costo operativo (COP) de revisar una alerta
COLUMNAS = ["monto", "hora", "distancia_km", "ingreso_mensual", "fraude"]


# =============================================================================
# PARTE A. ARCHIVO ALEATORIO INGENUO
# =============================================================================
def generar_ingenuo(n, rng):
    """Todo uniforme e independiente, fraude al 50 %: el 'primer intento'."""
    return pd.DataFrame({
        "monto": np.round(rng.uniform(1_000, 10_000_000, n), -2),
        "hora": rng.integers(0, 24, n),
        "distancia_km": np.round(rng.uniform(0, 1_000, n), 2),
        "ingreso_mensual": np.round(rng.uniform(1_000_000, 30_000_000, n), -3),
        "fraude": rng.integers(0, 2, n),
    })[COLUMNAS]


# =============================================================================
# PARTE C. ARCHIVO REALISTA
# =============================================================================
# Perfil horario de compras legítimas (peso relativo por hora 0..23):
# casi nada de madrugada, pico al almuerzo y en la tarde-noche.
PERFIL_HORARIO = np.array([0.8, 0.5, 0.3, 0.2, 0.2, 0.4, 1.2, 2.5, 3.8, 4.5, 5.0, 5.5,
                           6.8, 6.5, 5.2, 5.0, 5.2, 5.8, 6.6, 6.8, 6.0, 4.5, 3.0, 1.6])
PERFIL_HORARIO = PERFIL_HORARIO / PERFIL_HORARIO.sum()

PATRONES_FRAUDE = {  # nombre: proporción dentro de los fraudes
    "no_presencial_alto_valor": 0.45,
    "prueba_de_tarjeta": 0.25,
    "camuflado": 0.30,
}


def _lognormal(rng, mediana, sigma, n):
    return rng.lognormal(np.log(mediana), sigma, n)


def generar_realista(n, tasa_fraude, rng, ruido_etiqueta=0.002):
    """
    Genera transacciones con relaciones plausibles entre variables.
    Devuelve el DataFrame y el patrón que originó cada fila (para análisis).
    """
    # --- Clientes: ingreso log-normal con piso en el salario mínimo ---------
    n_clientes = max(n // 8, 100)
    ingresos_cli = np.maximum(SALARIO_MINIMO, _lognormal(rng, 3_200_000, 0.85, n_clientes))
    ingreso = ingresos_cli[rng.integers(0, n_clientes, n)]

    es_fraude = rng.random(n) < tasa_fraude
    patron = np.array(["legitima"] * n, dtype=object)
    monto = np.empty(n)
    hora = np.empty(n, dtype=int)
    dist = np.empty(n)

    # --- Transacciones legítimas -------------------------------------------
    L = ~es_fraude
    nl = L.sum()
    monto[L] = _lognormal(rng, 1, 1.0, nl) * 0.012 * ingreso[L]      # ~1.2 % del ingreso
    hora[L] = rng.choice(24, nl, p=PERFIL_HORARIO)
    # 1.5 % de compras grandes PLANIFICADAS y legítimas (electrodomésticos,
    # matrículas...): montos altos, pero de día y cerca de casa. Obligan al
    # modelo a combinar variables en lugar de mirar solo el monto.
    grande = rng.random(nl) < 0.015
    monto_l = monto[L]
    monto_l[grande] = ingreso[L][grande] * rng.uniform(0.2, 1.2, grande.sum())
    monto[L] = monto_l
    tipo = rng.random(nl)
    d = _lognormal(rng, 4, 0.9, nl)                                   # barrio / ciudad
    d = np.where(tipo > 0.90, rng.uniform(20, 150, nl), d)            # municipios cercanos
    d = np.where(tipo > 0.98, rng.uniform(200, 1500, nl), d)          # viajes
    d = np.where(grande, _lognormal(rng, 6, 0.8, nl), d)              # compra grande: cerca
    dist[L] = d
    h_l = hora[L]
    h_l[grande] = rng.choice(np.arange(9, 20), grande.sum())          # compra grande: de día
    hora[L] = h_l

    # --- Transacciones fraudulentas (mezcla de patrones) --------------------
    idx_f = np.flatnonzero(es_fraude)
    nombres = list(PATRONES_FRAUDE)
    asignado = rng.choice(nombres, len(idx_f), p=list(PATRONES_FRAUDE.values()))
    for nombre in nombres:
        ids = idx_f[asignado == nombre]
        k = len(ids)
        patron[ids] = nombre
        if nombre == "no_presencial_alto_valor":
            # compras grandes (30 %..250 % del ingreso), de madrugada, lejos
            monto[ids] = ingreso[ids] * rng.uniform(0.3, 2.5, k)
            noche = rng.random(k) < 0.55
            hora[ids] = np.where(noche, rng.integers(0, 6, k), rng.choice(24, k, p=PERFIL_HORARIO))
            dist[ids] = _lognormal(rng, 300, 1.0, k)
        elif nombre == "prueba_de_tarjeta":
            # montos pequeños para validar la tarjeta robada, a cualquier hora
            monto[ids] = rng.uniform(2_000, 20_000, k)
            hora[ids] = rng.integers(0, 24, k)
            dist[ids] = _lognormal(rng, 500, 0.8, k)
        else:
            # tarjeta robada usada cerca: montos medios-altos para SU ingreso,
            # con cierta preferencia por la noche (22 h - 5 h)
            monto[ids] = ingreso[ids] * rng.uniform(0.08, 0.6, k)
            noche = rng.random(k) < 0.40
            hora[ids] = np.where(noche, rng.choice([22, 23, 0, 1, 2, 3, 4, 5], k),
                                 rng.choice(24, k, p=PERFIL_HORARIO))
            dist[ids] = _lognormal(rng, 15, 1.0, k)

    # --- Ruido de etiqueta: contracargos erróneos / fraudes no reportados ---
    etiqueta = es_fraude.copy()
    voltear = rng.random(n) < ruido_etiqueta
    etiqueta[voltear] = ~etiqueta[voltear]

    df = pd.DataFrame({
        "monto": np.round(np.clip(monto, 1_000, None), -2),
        "hora": hora,
        "distancia_km": np.round(np.clip(dist, 0.05, 15_000), 2),
        "ingreso_mensual": np.round(ingreso, -3),
        "fraude": etiqueta.astype(int),
    })[COLUMNAS]
    return df, patron


# =============================================================================
# PARTE B. REVISIÓN CRÍTICA (diagnóstico cuantitativo)
# =============================================================================
def auc_univariada(x, y):
    """Poder discriminante de UNA variable (0.5 = nulo, 1 = perfecto)."""
    if y.min() == y.max():
        return float("nan")
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def diagnostico(df):
    y = df["fraude"].values
    ratio = df["monto"] / df["ingreso_mensual"]
    return OrderedDictLike([
        ("Tasa de fraude", f"{100 * y.mean():.2f} %"),
        ("Asimetría del monto", f"{df['monto'].skew():.2f}"),
        ("Mediana del monto (COP)", f"{df['monto'].median():,.0f}"),
        ("Transacciones con monto > ingreso", f"{100 * (ratio > 1).mean():.2f} %"),
        ("Compras entre 0 h y 5 h", f"{100 * (df['hora'] < 6).mean():.1f} %"),
        ("Mediana de distancia (km)", f"{df['distancia_km'].median():,.1f}"),
        ("Distancia > 100 km", f"{100 * (df['distancia_km'] > 100).mean():.1f} %"),
        ("Ingreso < salario mínimo", f"{100 * (df['ingreso_mensual'] < SALARIO_MINIMO).mean():.1f} %"),
        ("AUC univariada: monto", f"{auc_univariada(df['monto'], y):.3f}"),
        ("AUC univariada: hora", f"{auc_univariada(-np.cos(2 * np.pi * df['hora'] / 24), y):.3f}"),
        ("AUC univariada: distancia", f"{auc_univariada(df['distancia_km'], y):.3f}"),
        ("AUC univariada: ingreso", f"{auc_univariada(df['ingreso_mensual'], y):.3f}"),
        ("AUC univariada: monto/ingreso", f"{auc_univariada(ratio, y):.3f}"),
    ])


def OrderedDictLike(pares):
    from collections import OrderedDict
    return OrderedDict(pares)


PROBLEMAS_Y_CAMBIOS = [
    ("Fraude al 50 %: en tarjetas la tasa real es < 1-2 %.",
     "Tasa de fraude configurable, 2 % por defecto (--tasa-fraude)."),
    ("La etiqueta es independiente de las variables: no hay patrón que aprender.",
     "El fraude se genera con mecanismos propios (3 patrones) distintos a los legítimos."),
    ("Montos uniformes hasta 10 M: el gasto real es asimétrico (muchas compras pequeñas).",
     "Monto log-normal, proporcional al ingreso del cliente (~1.2 % en mediana)."),
    ("Monto sin relación con el ingreso: ~1/3 de las compras superan el ingreso mensual.",
     "El fraude de alto valor usa 30-250 % del ingreso; lo legítimo rara vez lo supera."),
    ("Hora uniforme: se compra igual a las 3 a.m. que a mediodía.",
     "Perfil horario diurno (picos 12-13 h y 18-20 h); fraude nocturno más frecuente."),
    ("Distancia uniforme 0-1000 km: casi nadie compra lejos de casa.",
     "Distancia log-normal (mediana ~4 km), 8 % municipios cercanos y 2 % viajes."),
    ("Ingresos uniformes 1-30 M: no refleja la distribución real (ni el salario mínimo).",
     "Ingreso log-normal por cliente (mediana 3.2 M) con piso en el salario mínimo."),
    ("Cada fila es un individuo distinto: no existen clientes recurrentes.",
     "Cada cliente realiza en promedio 8 transacciones con el mismo ingreso."),
    ("Etiquetas perfectas: en la realidad hay contracargos erróneos y fraudes no reportados.",
     "Ruido de etiqueta del 0.2 %."),
    ("Todos los fraudes se parecen (o no se parecen a nada).",
     "Fraude camuflado (30 %) que se confunde con compras normales: el problema no es trivial."),
    ("Todo monto alto sería fraude: en la realidad hay compras grandes legítimas.",
     "1.5 % de compras grandes planificadas (20-120 % del ingreso), de día y cerca de casa."),
]


# =============================================================================
# PARTE D. PROGRAMACIÓN GENÉTICA
# =============================================================================
TERMINALES = ["M", "I", "D", "HC", "HS"]


class Preprocesador:
    """Transforma las 4 variables crudas en los terminales del árbol."""

    def ajustar(self, df):
        crudo = self._crudo(df)
        self.media = {k: crudo[k].mean() for k in ("M", "I", "D")}
        self.desv = {k: crudo[k].std() + 1e-12 for k in ("M", "I", "D")}
        return self

    @staticmethod
    def _crudo(df):
        return {
            "M": np.log10(df["monto"].values.astype(float)),
            "I": np.log10(df["ingreso_mensual"].values.astype(float)),
            "D": np.log10(1 + df["distancia_km"].values.astype(float)),
            "HC": np.cos(2 * np.pi * df["hora"].values / 24),
            "HS": np.sin(2 * np.pi * df["hora"].values / 24),
        }

    def transformar(self, df):
        c = self._crudo(df)
        for k in ("M", "I", "D"):
            c[k] = (c[k] - self.media[k]) / self.desv[k]
        return [c[t] for t in TERMINALES]


# ---- Conjunto de funciones (vectorizadas con numpy; propiedad de clausura) --
def div_protegida(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-6, a / np.where(np.abs(b) > 1e-6, b, 1.0), 1.0)


def si_mayor(a, b, c, d):
    return np.where(a > b, c, d)


def construir_pset():
    pset = gp.PrimitiveSet("RIESGO", len(TERMINALES))
    pset.addPrimitive(np.add, 2, name="sum")
    pset.addPrimitive(np.subtract, 2, name="res")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(div_protegida, 2, name="div")
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    pset.addPrimitive(si_mayor, 4, name="si_mayor")
    pset.addEphemeralConstant("R", partial(lambda: round(random.uniform(-2, 2), 2)))
    pset.renameArguments(**{f"ARG{i}": t for i, t in enumerate(TERMINALES)})
    return pset


def puntaje(individuo, pset, X):
    """Ejecuta el programa-árbol sobre todas las filas a la vez."""
    f = gp.compile(individuo, pset)
    with np.errstate(all="ignore"):
        s = f(*X)
    s = np.broadcast_to(np.asarray(s, dtype=float), X[0].shape)
    return np.clip(np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def auc_rapida(y, s):
    """AUC-ROC por el estadístico de Mann-Whitney (con manejo de empates)."""
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluar(individuo, pset, X, y, modo, lam):
    s = puntaje(individuo, pset, X)
    if modo == "auc":
        return (auc_rapida(y, s) - lam * len(individuo),)
    # f_apt del documento con error absoluto ponderado por clase
    p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    E = np.sum(w * np.abs(y - p)) / np.sum(w)
    return (1 / (0.1 + E) - lam * len(individuo),)


if not hasattr(creator, "AptitudFraude"):
    creator.create("AptitudFraude", base.Fitness, weights=(1.0,))
    creator.create("ArbolFraude", gp.PrimitiveTree, fitness=creator.AptitudFraude)


def mutacion_mixta(ind, expr, pset):
    r = random.random()
    if r < 0.4:
        return gp.mutUniform(ind, expr=expr, pset=pset)
    if r < 0.6:
        return gp.mutNodeReplacement(ind, pset=pset)
    if r < 0.85:
        return gp.mutEphemeral(ind, mode="one")     # ajusta una constante
    return gp.mutShrink(ind)


def correr_pg(X_tr, y_tr, X_va, y_va, args, semilla, verbose=True, etiqueta=""):
    random.seed(semilla)
    np.random.seed(semilla)
    pset = construir_pset()
    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individuo", tools.initIterate, creator.ArbolFraude, tb.expr)
    tb.register("poblacion", tools.initRepeat, list, tb.individuo)
    tb.register("evaluate", evaluar, pset=pset, X=X_tr, y=y_tr, modo=args.aptitud, lam=args.parsimonia)
    tb.register("select", tools.selDoubleTournament, fitness_size=args.torneo,
                parsimony_size=args.presion_tamano, fitness_first=True)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genGrow, min_=0, max_=3)
    tb.register("mutate", mutacion_mixta, expr=tb.expr_mut, pset=pset)
    limite = gp.staticLimit(key=operator.attrgetter("height"), max_value=args.altura_max)
    tb.decorate("mate", limite)
    tb.decorate("mutate", limite)

    pob = tb.poblacion(n=args.poblacion)
    for ind in pob:
        ind.fitness.values = tb.evaluate(ind)

    hist = {"gen": [], "apt_mejor": [], "apt_media": [], "auc_tr": [], "auc_va": [], "tam_medio": []}
    designado, auc_va_designado = None, -1
    for gen in range(args.generaciones + 1):
        elite = tools.selBest(pob, args.elitismo)
        # Designación (paso 6): entre la élite, el de mejor AUC en validación
        for cand in elite:
            a_va = auc_rapida(y_va, puntaje(cand, pset, X_va))
            if a_va > auc_va_designado + 1e-9 or (abs(a_va - auc_va_designado) <= 1e-9
                                                 and len(cand) < len(designado)):
                designado, auc_va_designado = tb.clone(cand), a_va
        apts = [i.fitness.values[0] for i in pob]
        hist["gen"].append(gen)
        hist["apt_mejor"].append(elite[0].fitness.values[0])
        hist["apt_media"].append(float(np.mean(apts)))
        hist["auc_tr"].append(auc_rapida(y_tr, puntaje(elite[0], pset, X_tr)))
        hist["auc_va"].append(auc_rapida(y_va, puntaje(elite[0], pset, X_va)))
        hist["tam_medio"].append(float(np.mean([len(i) for i in pob])))
        if verbose and (gen % 10 == 0 or gen == args.generaciones):
            print(f"   {etiqueta}gen {gen:3d}  aptitud={elite[0].fitness.values[0]:.4f}  "
                  f"AUC ent={hist['auc_tr'][-1]:.4f}  AUC val={hist['auc_va'][-1]:.4f}  "
                  f"nodos mejor={len(elite[0]):3d}  nodos medio={hist['tam_medio'][-1]:.1f}")
        if gen == args.generaciones:
            break
        hijos = tb.select(pob, len(pob) - args.elitismo)
        hijos = [tb.clone(h) for h in hijos]
        for i in range(1, len(hijos), 2):
            if random.random() < args.pc:
                hijos[i - 1], hijos[i] = tb.mate(hijos[i - 1], hijos[i])
                del hijos[i - 1].fitness.values, hijos[i].fitness.values
        for i in range(len(hijos)):
            if random.random() < args.pm:
                (hijos[i],) = tb.mutate(hijos[i])
                del hijos[i].fitness.values
        for h in hijos:
            if not h.fitness.valid:
                h.fitness.values = tb.evaluate(h)
        pob = hijos + [tb.clone(e) for e in elite]
    return designado, pset, hist


# =============================================================================
# Traducción del árbol a expresión legible y a código Python
# =============================================================================
def a_expresion(individuo, estilo="texto"):
    """Convierte la lista prefija de DEAP a notación infija."""
    pila = []
    for nodo in reversed(individuo):
        if isinstance(nodo, gp.Primitive):
            args_ = [pila.pop() for _ in range(nodo.arity)]
            n = nodo.name
            if n == "sum":
                e = f"({args_[0]} + {args_[1]})"
            elif n == "res":
                e = f"({args_[0]} - {args_[1]})"
            elif n == "mul":
                e = f"({args_[0]} * {args_[1]})"
            elif n == "neg":
                e = f"(-{args_[0]})"
            elif n == "div":
                e = f"div({args_[0]}, {args_[1]})"
            elif n == "si_mayor":
                e = (f"({args_[2]} if {args_[0]} > {args_[1]} else {args_[3]})" if estilo == "python"
                     else f"si({args_[0]} > {args_[1]}; {args_[2]}; {args_[3]})")
            else:
                e = f"{n}({', '.join(args_)})"
            pila.append(e)
        else:
            v = nodo.value
            pila.append(f"{v:.4g}" if isinstance(v, float) else str(v))
    return pila[0]


def exportar_modelo(individuo, prep, umbral, ruta, metricas):
    expr = a_expresion(individuo, "python")
    codigo = f'''# -*- coding: utf-8 -*-
"""
Modelo de detección de fraude obtenido por Programación Genética (DEAP).
Generado automáticamente por ej3_pg_deteccion_fraudes.py

Métricas en el conjunto de PRUEBA:
{metricas}
"""
import math

UMBRAL = {float(umbral)!r}
_MEDIA = {{"M": {float(prep.media["M"])!r}, "I": {float(prep.media["I"])!r}, "D": {float(prep.media["D"])!r}}}
_DESV = {{"M": {float(prep.desv["M"])!r}, "I": {float(prep.desv["I"])!r}, "D": {float(prep.desv["D"])!r}}}


def div(a, b):
    return a / b if abs(b) > 1e-6 else 1.0


def puntaje_riesgo(monto, hora, distancia_km, ingreso_mensual):
    """Puntaje de riesgo: mayor valor = más sospechosa la transacción."""
    M = (math.log10(monto) - _MEDIA["M"]) / _DESV["M"]
    I = (math.log10(ingreso_mensual) - _MEDIA["I"]) / _DESV["I"]
    D = (math.log10(1 + distancia_km) - _MEDIA["D"]) / _DESV["D"]
    HC = math.cos(2 * math.pi * hora / 24)
    HS = math.sin(2 * math.pi * hora / 24)
    return {expr}


def es_fraude(monto, hora, distancia_km, ingreso_mensual):
    """True si la transacción debe marcarse como posible fraude."""
    return puntaje_riesgo(monto, hora, distancia_km, ingreso_mensual) >= UMBRAL


if __name__ == "__main__":
    ejemplos = [
        ("Almuerzo cerca de casa", 35_000, 13, 2.5, 3_000_000),
        ("Mercado del mes", 450_000, 18, 5.0, 4_500_000),
        ("Compra de 6 M a las 3 a.m. a 800 km", 6_000_000, 3, 800, 3_000_000),
        ("Cobro de 5.000 a 600 km (prueba de tarjeta)", 5_000, 16, 600, 2_500_000),
        ("Viaje: hotel en otra ciudad", 380_000, 21, 420, 9_000_000),
    ]
    for desc, m, h, d, i in ejemplos:
        s = puntaje_riesgo(m, h, d, i)
        print(f"{{desc:<45}} puntaje={{s:9.3f}}  ->  {{'POSIBLE FRAUDE' if s >= UMBRAL else 'normal'}}")
'''
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(codigo)


# =============================================================================
# PARTE E. EVALUACIÓN
# =============================================================================
def umbral_optimo_f1(y, s):
    prec, rec, umb = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1[:-1]))
    return float(umb[k])


def metricas(nombre, y, s, umbral, montos):
    pred = (s >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    perdida = montos[(y == 1) & (pred == 0)].sum()
    costo = perdida + COSTO_REVISION * (tp + fp)
    return {
        "modelo": nombre, "AUC-ROC": roc_auc_score(y, s), "AUC-PR": average_precision_score(y, s),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "costo_MCOP": costo / 1e6,
    }


def importancia_permutacion(ind, pset, X, y, rng, repeticiones=5):
    base_auc = auc_rapida(y, puntaje(ind, pset, X))
    res = {}
    for k, t in enumerate(TERMINALES):
        caidas = []
        for _ in range(repeticiones):
            Xp = [c.copy() for c in X]
            Xp[k] = rng.permutation(Xp[k])
            caidas.append(base_auc - auc_rapida(y, puntaje(ind, pset, Xp)))
        res[t] = float(np.mean(caidas))
    return res


# =============================================================================
# GRÁFICAS
# =============================================================================
def graficar_distribuciones(df_i, df_r, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(2, 5, figsize=(22, 8))
    for fila, (df, titulo) in enumerate(((df_i, "INGENUO"), (df_r, "REALISTA"))):
        leg, fra = df[df.fraude == 0], df[df.fraude == 1]
        specs = [
            ("monto", "log10(monto COP)", lambda d: np.log10(d["monto"])),
            ("hora", "hora del día", lambda d: d["hora"]),
            ("distancia", "log10(1 + distancia km)", lambda d: np.log10(1 + d["distancia_km"])),
            ("ingreso", "log10(ingreso mensual)", lambda d: np.log10(d["ingreso_mensual"])),
            ("ratio", "log10(monto / ingreso)", lambda d: np.log10(d["monto"] / d["ingreso_mensual"])),
        ]
        for c, (_, etiqueta, f) in enumerate(specs):
            ax = ejes[fila, c]
            bins = 24 if etiqueta == "hora del día" else 40
            ax.hist(f(leg), bins=bins, density=True, alpha=0.55, label="legítima", color="tab:blue")
            ax.hist(f(fra), bins=bins, density=True, alpha=0.55, label="fraude", color="tab:red")
            ax.set_xlabel(etiqueta)
            if c == 0:
                ax.set_ylabel(f"{titulo}\ndensidad")
                ax.legend()
    fig.suptitle("Distribuciones por clase: archivo ingenuo (arriba) vs. realista (abajo)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_convergencia(hist, hist_ing, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(hist["gen"], hist["auc_tr"], label="AUC entrenamiento (realista)", lw=2)
    a1.plot(hist["gen"], hist["auc_va"], label="AUC validación (realista)", lw=2, ls="--")
    a1.plot(hist_ing["gen"], hist_ing["auc_va"], label="AUC validación (ingenuo)", lw=2, color="gray")
    a1.axhline(0.5, color="k", lw=0.8, ls=":")
    a1.set_xlabel("generación")
    a1.set_ylabel("AUC-ROC del mejor individuo")
    a1.set_ylim(0.4, 1.0)
    a1.legend()
    a1.set_title("Historia de la aptitud")
    a2.plot(hist["gen"], hist["tam_medio"], color="tab:purple")
    a2.set_xlabel("generación")
    a2.set_ylabel("nodos promedio en la población")
    a2.set_title("Tamaño de los árboles (control de 'bloat')")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_roc_pr(y, puntajes, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for nombre, s in puntajes.items():
        fpr, tpr, _ = roc_curve(y, s)
        a1.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC={roc_auc_score(y, s):.3f})")
        p, r, _ = precision_recall_curve(y, s)
        a2.plot(r, p, lw=2, label=f"{nombre} (AP={average_precision_score(y, s):.3f})")
    a1.plot([0, 1], [0, 1], "k:", lw=1)
    a1.set_xlabel("tasa de falsos positivos")
    a1.set_ylabel("tasa de verdaderos positivos (recall)")
    a1.set_title("Curva ROC - conjunto de prueba")
    a1.legend(loc="lower right")
    a2.axhline(y.mean(), color="k", ls=":", lw=1, label=f"azar ({y.mean():.3f})")
    a2.set_xlabel("recall (fraudes detectados)")
    a2.set_ylabel("precisión (alertas que son fraude)")
    a2.set_title("Curva precisión-recall - conjunto de prueba")
    a2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
class Tee:
    def __init__(self, *f):
        self.f = f

    def write(self, t):
        for x in self.f:
            x.write(t)

    def flush(self):
        for x in self.f:
            x.flush()


def particionar(df, semilla):
    tr, tmp = train_test_split(df, test_size=0.4, stratify=df["fraude"], random_state=semilla)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["fraude"], random_state=semilla)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def tabla(filas, columnas, formatos):
    enc = " ".join(f"{c:>{w}}" for c, (w, _) in zip(columnas, formatos))
    print(" " + enc)
    for f in filas:
        print(" " + " ".join(f"{f[c]:>{w}{fmt}}" for c, (w, fmt) in zip(columnas, formatos)))


def main():
    p = argparse.ArgumentParser(description="PG para detección de fraudes")
    p.add_argument("--n", type=int, default=20_000, help="número de transacciones")
    p.add_argument("--tasa-fraude", dest="tasa_fraude", type=float, default=0.02)
    p.add_argument("--aptitud", choices=["auc", "error"], default="auc")
    p.add_argument("--parsimonia", type=float, default=0.0002, help="penalización por nodo")
    p.add_argument("--poblacion", type=int, default=400)
    p.add_argument("--generaciones", type=int, default=60)
    p.add_argument("--pc", type=float, default=0.7)
    p.add_argument("--pm", type=float, default=0.3)
    p.add_argument("--elitismo", type=int, default=3)
    p.add_argument("--torneo", type=int, default=3)
    p.add_argument("--presion-tamano", dest="presion_tamano", type=float, default=1.2,
                   help="tamaño del torneo de parsimonia (1 = sin presión, 2 = máxima)")
    p.add_argument("--altura-max", dest="altura_max", type=int, default=8)
    p.add_argument("--corridas", type=int, default=3,
                   help="corridas independientes de PG; se designa la de mejor validación")
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--salida", default="resultados_ej3")
    args = p.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    reporte = open(os.path.join(args.salida, "reporte_ej3.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, reporte)
    t0 = time.time()
    rng = np.random.default_rng(args.semilla)
    pd.set_option("display.width", 140)

    # ------------------------------------------------------------------ A
    print("=" * 90)
    print(" PARTE A. ARCHIVO ALEATORIO INGENUO")
    print("=" * 90)
    df_ing = generar_ingenuo(args.n, rng)
    ruta_ing = os.path.join(args.salida, "transacciones_ingenuo.csv")
    df_ing.to_csv(ruta_ing, index=False)
    print(f" Guardado: {ruta_ing}  ({len(df_ing)} filas)")
    print(df_ing.head(8).to_string(index=False))

    # ------------------------------------------------------------------ C (se genera antes de B para comparar)
    df_real, patron = generar_realista(args.n, args.tasa_fraude, rng)
    ruta_real = os.path.join(args.salida, "transacciones_realista.csv")
    df_real.to_csv(ruta_real, index=False)

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 90)
    print(" PARTE B. REVISIÓN CRÍTICA DEL ARCHIVO INGENUO Y CAMBIOS REALIZADOS")
    print("=" * 90)
    d_i, d_r = diagnostico(df_ing), diagnostico(df_real)
    print(f" {'Indicador':<38}{'Ingenuo':>16}{'Realista':>16}")
    for k in d_i:
        print(f" {k:<38}{d_i[k]:>16}{d_r[k]:>16}")
    print("\n Problemas detectados  ->  cambio aplicado")
    for i, (prob, cambio) in enumerate(PROBLEMAS_Y_CAMBIOS, 1):
        print(f" {i:2d}. {prob}\n       -> {cambio}")

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 90)
    print(" PARTE C. ARCHIVO REALISTA")
    print("=" * 90)
    print(f" Guardado: {ruta_real}  ({len(df_real)} filas, {df_real.fraude.sum()} fraudes)")
    print(df_real.sample(8, random_state=1).to_string(index=False))
    print("\n Fraudes por patrón de generación (antes del ruido de etiqueta):")
    for nombre, cnt in pd.Series(patron[patron != "legitima"]).value_counts().items():
        print(f"   {nombre:<28} {cnt:5d}")
    print("\n Estadísticas descriptivas por clase (medianas):")
    print(df_real.groupby("fraude").median().to_string())
    graficar_distribuciones(df_ing, df_real, os.path.join(args.salida, "distribuciones_ingenuo_vs_realista.png"))

    # ------------------------------------------------------------------ D
    print("\n" + "=" * 90)
    print(" PARTE D. MODELO POR PROGRAMACIÓN GENÉTICA")
    print("=" * 90)
    print(f" Terminales T : {TERMINALES} + constantes efímeras R en [-2, 2]")
    print(" Funciones  F : +, -, *, div protegida, neg, max, min, si_mayor(a,b,c,d)")
    print(f" Aptitud      : {'AUC-ROC(entrenamiento) - λ·nodos' if args.aptitud == 'auc' else 'f_apt = 1/(0.1 + E_ponderado) - λ·nodos'}"
          f"  (λ = {args.parsimonia})")
    print(f" Parámetros   : pobl={args.poblacion}, gen={args.generaciones}, pc={args.pc}, pm={args.pm}, "
          f"elitismo={args.elitismo}, altura≤{args.altura_max}")

    # D.1  PG sobre el archivo ingenuo (control)
    print("\n D.1  Control: PG sobre el archivo INGENUO (no debería aprender nada)")
    tr_i, va_i, te_i = particionar(df_ing, args.semilla)
    prep_i = Preprocesador().ajustar(tr_i)
    args_ctrl = argparse.Namespace(**{**vars(args), "generaciones": min(20, args.generaciones)})
    mejor_i, pset, hist_i = correr_pg(prep_i.transformar(tr_i), tr_i.fraude.values,
                                      prep_i.transformar(va_i), va_i.fraude.values,
                                      args_ctrl, args.semilla, etiqueta="[ingenuo] ")
    auc_te_i = auc_rapida(te_i.fraude.values, puntaje(mejor_i, pset, prep_i.transformar(te_i)))
    print(f"   AUC en prueba del mejor modelo (ingenuo): {auc_te_i:.4f}  ->  equivalente al azar")

    # D.2  PG sobre el archivo realista
    print("\n D.2  PG sobre el archivo REALISTA")
    tr, va, te = particionar(df_real, args.semilla)
    print(f"   Partición estratificada: entrenamiento={len(tr)} ({tr.fraude.sum()} fraudes), "
          f"validación={len(va)} ({va.fraude.sum()}), prueba={len(te)} ({te.fraude.sum()})")
    prep = Preprocesador().ajustar(tr)
    X_tr, X_va, X_te = prep.transformar(tr), prep.transformar(va), prep.transformar(te)
    y_tr, y_va, y_te = tr.fraude.values, va.fraude.values, te.fraude.values
    mejor, auc_va_mejor, hist = None, -1, None
    for c in range(args.corridas):
        if args.corridas > 1:
            print(f"   --- corrida independiente {c + 1}/{args.corridas}")
        cand, pset, h = correr_pg(X_tr, y_tr, X_va, y_va, args, args.semilla + 1 + 101 * c)
        a_va = auc_rapida(y_va, puntaje(cand, pset, X_va))
        if a_va > auc_va_mejor:
            mejor, auc_va_mejor, hist = cand, a_va, h
    if args.corridas > 1:
        print(f"   Se designa el modelo de la corrida con mayor AUC de validación ({auc_va_mejor:.4f})")

    print("\n MODELO DESIGNADO (mejor AUC en validación entre la élite de cada generación)")
    print(f"   Nodos: {len(mejor)}   Altura: {mejor.height}")
    print(f"   Programa (prefijo DEAP): {mejor}")
    print(f"   Puntaje de riesgo s(x) = {a_expresion(mejor)}")
    print("   donde M, I, D son log10(monto), log10(ingreso), log10(1+dist) estandarizados")
    print(f"   (medias {prep.media['M']:.3f}, {prep.media['I']:.3f}, {prep.media['D']:.3f};"
          f" desv. {prep.desv['M']:.3f}, {prep.desv['I']:.3f}, {prep.desv['D']:.3f}),"
          f" HC = cos(2πh/24), HS = sin(2πh/24).")

    # ------------------------------------------------------------------ E
    print("\n" + "=" * 90)
    print(" PARTE E. EVALUACIÓN EN EL CONJUNTO DE PRUEBA (datos nunca vistos)")
    print("=" * 90)
    s_va, s_te = puntaje(mejor, pset, X_va), puntaje(mejor, pset, X_te)
    umbral = umbral_optimo_f1(y_va, s_va)
    print(f" Umbral de decisión (máx. F1 en validación): {umbral:.4f}")

    # Modelos de referencia con las mismas variables transformadas
    M_tr, M_va, M_te = (np.column_stack(X) for X in (X_tr, X_va, X_te))
    rl = LogisticRegression(class_weight="balanced", max_iter=2000).fit(M_tr, y_tr)
    ad = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0).fit(M_tr, y_tr)
    modelos = {
        "PG (DEAP)": (s_va, s_te),
        "Regresión logística": (rl.decision_function(M_va), rl.decision_function(M_te)),
        "Árbol decisión (prof. 4)": (ad.predict_proba(M_va)[:, 1], ad.predict_proba(M_te)[:, 1]),
    }
    montos_te = te["monto"].values
    filas = [metricas("Sin modelo (nunca fraude)", y_te, np.zeros(len(y_te)), 0.5, montos_te)]
    for nombre, (sv, st) in modelos.items():
        filas.append(metricas(nombre, y_te, st, umbral_optimo_f1(y_va, sv), montos_te))
    cols = ["modelo", "AUC-ROC", "AUC-PR", "precision", "recall", "F1", "TP", "FP", "FN", "costo_MCOP"]
    fmts = [(26, ""), (8, ".4f"), (7, ".4f"), (9, ".3f"), (7, ".3f"), (6, ".3f"), (5, "d"), (5, "d"),
            (5, "d"), (11, ".1f")]
    for f in filas:
        f["modelo"] = f["modelo"][:26]
    tabla(filas, cols, fmts)
    print(f" costo_MCOP = montos de fraudes NO detectados + {COSTO_REVISION:,} COP por alerta revisada"
          " (millones de COP)")

    fg = filas[1]
    print("\n Matriz de confusión del modelo PG (prueba):")
    print("                     predicho legítima   predicho fraude")
    print(f"   real legítima     {fg['TN']:>17d} {fg['FP']:>17d}")
    print(f"   real fraude       {fg['FN']:>17d} {fg['TP']:>17d}")

    # Detección por patrón de fraude (solo diagnóstico; el modelo no ve el patrón)
    _, _, te_idx = particionar(pd.DataFrame({"i": np.arange(len(df_real)), "fraude": df_real.fraude}),
                               args.semilla)
    patron_te = patron[te_idx["i"].values]
    pred_te = s_te >= umbral
    print("\n Recall del modelo PG por patrón de fraude (prueba):")
    for nombre in PATRONES_FRAUDE:
        mask = (patron_te == nombre) & (y_te == 1)
        if mask.sum():
            print(f"   {nombre:<28} {pred_te[mask].mean():6.1%}  ({mask.sum()} casos)")
    ruido = (patron_te == "legitima") & (y_te == 1)
    if ruido.sum():
        print(f"   {'(ruido de etiqueta)':<28} {pred_te[ruido].mean():6.1%}  ({ruido.sum()} casos: "
              "legítimas marcadas como fraude)")

    print("\n Importancia por permutación (caída de AUC al barajar cada terminal):")
    imp = importancia_permutacion(mejor, pset, X_te, y_te, np.random.default_rng(0))
    nombres_largos = {"M": "monto", "I": "ingreso", "D": "distancia", "HC": "hora (cos)", "HS": "hora (sin)"}
    for t, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"   {t:<3} {nombres_largos[t]:<12} {v:7.4f}  {'#' * int(max(v, 0) * 200)}")

    print("\n Reglas del árbol de decisión de referencia (para comparar interpretabilidad):")
    print("   " + export_text(ad, feature_names=TERMINALES, decimals=2).replace("\n", "\n   "))

    # ------------------------------------------------------------------ F
    texto_met = (f"  AUC-ROC={fg['AUC-ROC']:.4f}  AUC-PR={fg['AUC-PR']:.4f}  precision={fg['precision']:.3f}"
                 f"  recall={fg['recall']:.3f}  F1={fg['F1']:.3f}")
    ruta_mod = os.path.join(args.salida, "modelo_fraude_pg.py")
    exportar_modelo(mejor, prep, umbral, ruta_mod, texto_met)
    # Verificación: el módulo exportado reproduce los puntajes del árbol
    espacio = {}
    exec(open(ruta_mod, encoding="utf-8").read(), espacio)
    muestra = te.head(200)
    s_exp = np.array([espacio["puntaje_riesgo"](r.monto, r.hora, r.distancia_km, r.ingreso_mensual)
                      for r in muestra.itertuples()])
    assert np.allclose(s_exp, s_te[:200], rtol=1e-6, atol=1e-6), "el modelo exportado no coincide"
    print(f"\n PARTE F. Modelo exportado a {ruta_mod} (verificado contra el árbol: OK)")

    graficar_convergencia(hist, hist_i, os.path.join(args.salida, "convergencia_pg_fraude.png"))
    graficar_roc_pr(y_te, {k: v[1] for k, v in modelos.items()}, os.path.join(args.salida, "curvas_roc_pr.png"))

    print(f"\n Archivos generados en {os.path.abspath(args.salida)}   (tiempo total {time.time() - t0:.0f} s)")
    sys.stdout = sys.__stdout__
    reporte.close()


if __name__ == "__main__":
    main()
