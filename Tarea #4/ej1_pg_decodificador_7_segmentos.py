"""
Librería usada: DEAP (Distributed Evolutionary Algorithms in Python)
    pip install deap numpy matplotlib

Ejecución
---------
    python ej1_pg_decodificador_7_segmentos.py                 # corrida estándar
    python ej1_pg_decodificador_7_segmentos.py --compuertas nand
    python ej1_pg_decodificador_7_segmentos.py --modo completo  # sin "don't care"
    python ej1_pg_decodificador_7_segmentos.py --comparar-conjuntos 5

Salidas (carpeta ./resultados_ej1):
    reporte_ej1.txt                  - todo lo impreso en consola
    convergencia_segmentos.png       - historia de la aptitud (fig. 4.12 del texto)
    arboles_segmentos.png            - los 7 árboles-programa evolucionados
    display_evolucionado.png         - dígitos 0..9 dibujados por el circuito
    decodificador_7seg_pg.v          - circuito en Verilog (listo para FPGA)
    decodificador_7seg_pg.py         - circuito como módulo Python autónomo

-------------------------------------------------------------------------------
 LOS SEIS PASOS PREPARATORIOS DE LA PG (sección 4.2 del documento)
-------------------------------------------------------------------------------
 1. PROBLEMA
    Un display de 7 segmentos (a..g) muestra un dígito decimal. El circuito
    decodificador recibe el dígito en código BCD (4 bits: A B C D, con A el
    bit más significativo) y enciende los segmentos correctos:

              a
            -----
         f |     | b          Dígito  A B C D   a b c d e f g
           |  g  |              0     0 0 0 0   1 1 1 1 1 1 0
            -----               1     0 0 0 1   0 1 1 0 0 0 0
         e |     | c            2     0 0 1 0   1 1 0 1 1 0 1
           |  d  |              ...   (tabla completa en TABLA_7SEG)
            -----

    Es un problema de 4 entradas y 7 salidas. Se evoluciona un árbol-programa
    (una expresión booleana) por cada segmento. Las combinaciones 10..15 no
    son BCD válidas: en modo "dontcare" (por defecto) se ignoran, como hace un
    diseñador con mapas de Karnaugh; en modo "completo" se exige display
    apagado para ellas.

 2. CONJUNTO DE TERMINALES   T = {A, B, C, D}
    Son las hojas del árbol: los 4 bits de entrada. No se incluyen las
    constantes 0/1 porque en lógica combinacional no aportan (x AND 1 = x)
    y solo agrandan el espacio de búsqueda.

 3. CONJUNTO DE FUNCIONES    F = compuertas lógicas
      - "basico"    : {AND, OR, NOT}              (conjunto clásico, completo)
      - "extendido" : {AND, OR, NOT, XOR}         (por defecto)
      - "nand"      : {NAND}                      (compuerta universal)
      - "universal" : {AND, OR, NOT, XOR, NAND, NOR}
    Propiedad de CLAUSURA: toda compuerta recibe bits y devuelve un bit, así
    que cualquier composición es un circuito válido. Propiedad de SUFICIENCIA:
    los tres conjuntos son funcionalmente completos (pueden expresar cualquier
    función booleana).

 4. MEDIDA DE APTITUD
    Se usa la misma forma que el documento (sección 4.9):

                 f_apt = 1 / (0.1 + SUMA_j | y_j - yc_j | )

    donde y_j es el valor deseado del segmento para la entrada j de la tabla
    de verdad e yc_j el valor que calcula el árbol. El máximo es 10 (error 0).
    Como objetivo SECUNDARIO se minimiza el número de compuertas: entre dos
    circuitos igual de correctos gana el más barato. Se aplica mediante
    selección por doble torneo (primero aptitud, luego tamaño) y en el
    elitismo/designación del resultado (orden lexicográfico errores ->
    compuertas). Esto controla el crecimiento desmedido de los árboles
    ("bloat") que el documento advierte en la sección 4.3.1 y la figura 4.12.

    Detalle de implementación (evaluación "bit-paralela"): cada terminal se
    representa como un entero de 16 bits cuyo bit n es el valor de la entrada
    para la combinación n (0..15). Así, UNA sola ejecución del árbol con
    operadores bit a bit (&, |, ^, ~) evalúa las 16 filas de la tabla de
    verdad a la vez; el error es el número de bits distintos (popcount).

 5. PARÁMETROS DE CONTROL (valores inspirados en el documento, sección 4.9)
      población = 300, generaciones máx. = 200, prob. cruce = 0.7,
      prob. mutación = 0.2 (por individuo), doble torneo (4 / 1.2), elitismo = 2,
      altura máxima del árbol = 8, inicialización "ramped half-and-half".

 6. DESIGNACIÓN DEL RESULTADO Y TERMINACIÓN
    El resultado es el mejor individuo (elitismo + Salón de la Fama).
    Se termina cuando se alcanza el máximo de generaciones, o bien
    'refinamiento' generaciones después de encontrar un circuito perfecto
    (esas generaciones extra sirven para reducir compuertas).

Operadores genéticos (sección 4.7 del documento):
    - Cruce de subárboles en un punto (gp.cxOnePoint) -> intercambia subárboles.
    - Mutación mixta:
        * de subárbol   (gp.mutUniform): reemplaza un subárbol por uno nuevo;
        * puntual       (gp.mutNodeReplacement): cambia una función por otra de
                         igual aridad o un terminal por otro terminal, como pide
                         el documento para preservar la clausura;
        * de encogimiento (gp.mutShrink): sustituye un subárbol por uno de sus
                         hijos -> favorece circuitos pequeños.
===============================================================================
"""

import argparse
import operator
import os
import random
import sys
import time
from collections import OrderedDict

try:
    from deap import base, creator, gp, tools
except ImportError:  # mensaje claro si falta la librería
    sys.exit("Falta la librería DEAP. Instálela con:  pip install deap")

import numpy as np

# =============================================================================
# 1. EL PROBLEMA: tabla de verdad del decodificador BCD -> 7 segmentos
# =============================================================================
SEGMENTOS = "abcdefg"

# Display de cátodo común (1 = segmento encendido). El 6 y el 9 llevan "cola".
TABLA_7SEG = {
    0: "1111110",
    1: "0110000",
    2: "1101101",
    3: "1111001",
    4: "0110011",
    5: "1011011",
    6: "1011111",
    7: "1110000",
    8: "1111111",
    9: "1111011",
}

N_FILAS = 16                      # 2^4 combinaciones de entrada
MASCARA = (1 << N_FILAS) - 1      # 0xFFFF

# Entradas como vectores de 16 bits (bit n = valor de la variable en la fila n)
# A es el bit más significativo (peso 8), D el menos significativo (peso 1).
ENTRADAS = OrderedDict(
    (nombre, sum(1 << n for n in range(N_FILAS) if (n >> peso) & 1))
    for nombre, peso in (("A", 3), ("B", 2), ("C", 1), ("D", 0))
)


def objetivo_segmento(seg):
    """Vector de 16 bits con el valor deseado del segmento en cada fila."""
    k = SEGMENTOS.index(seg)
    return sum(1 << n for n, patron in TABLA_7SEG.items() if patron[k] == "1")


# Valores de hojas para el árbol post-procesado (incluye constantes que puede
# introducir la simplificación algebraica).
VALORES_HOJAS = dict(ENTRADAS, **{"0": 0, "1": MASCARA})


def mascara_cuidado(modo):
    """Filas que cuentan en la aptitud. 'dontcare': solo 0..9; 'completo': 0..15."""
    return MASCARA if modo == "completo" else sum(1 << n for n in range(10))


def contar_unos(x):
    return bin(x).count("1")


# =============================================================================
# 3. CONJUNTO DE FUNCIONES (compuertas) - versión bit-paralela
# =============================================================================
def AND(x, y):
    return x & y


def OR(x, y):
    return x | y


def XOR(x, y):
    return x ^ y


def NOT(x):
    return ~x & MASCARA


def NAND(x, y):
    return ~(x & y) & MASCARA


def NOR(x, y):
    return ~(x | y) & MASCARA


CONJUNTOS_FUNCIONES = {
    "basico": [("AND", AND, 2), ("OR", OR, 2), ("NOT", NOT, 1)],
    "extendido": [("AND", AND, 2), ("OR", OR, 2), ("NOT", NOT, 1), ("XOR", XOR, 2)],
    "nand": [("NAND", NAND, 2)],
    "universal": [("AND", AND, 2), ("OR", OR, 2), ("NOT", NOT, 1), ("XOR", XOR, 2),
                  ("NAND", NAND, 2), ("NOR", NOR, 2)],
}
CONMUTATIVAS = {"AND", "OR", "XOR", "NAND", "NOR"}
FUNC_POR_NOMBRE = {"AND": AND, "OR": OR, "XOR": XOR, "NOT": NOT, "NAND": NAND, "NOR": NOR}


def construir_pset(conjunto):
    """Pasos 2 y 3: conjunto de terminales T y de funciones F para DEAP."""
    pset = gp.PrimitiveSet("CIRCUITO", len(ENTRADAS))
    for nombre, func, aridad in CONJUNTOS_FUNCIONES[conjunto]:
        pset.addPrimitive(func, aridad, name=nombre)
    pset.renameArguments(**{f"ARG{i}": n for i, n in enumerate(ENTRADAS)})
    return pset


# =============================================================================
# 4. FUNCIÓN DE APTITUD
# =============================================================================
def n_compuertas(individuo):
    return sum(isinstance(nodo, gp.Primitive) for nodo in individuo)


def evaluar(individuo, pset, objetivo, cuidado):
    """
    f_apt = 1 / (0.1 + errores)  (ecuación de la sección 4.9 del documento)
    """
    programa = gp.compile(individuo, pset)          # árbol -> función Python
    salida = programa(*ENTRADAS.values())           # ejecuta las 16 filas a la vez
    errores = contar_unos((salida ^ objetivo) & cuidado)
    return (1.0 / (0.1 + errores),)


def errores_de(individuo):
    """Recupera el número de errores a partir de f_apt."""
    return int(round(1.0 / individuo.fitness.values[0] - 0.1))


def clave(individuo):
    """Orden lexicográfico: primero menos errores, luego menos compuertas."""
    return (errores_de(individuo), n_compuertas(individuo), len(individuo))


# =============================================================================
# 5. MOTOR DE PG (DEAP)
# =============================================================================
# Aptitud escalar (se maximiza f_apt). La parsimonia se aplica en la SELECCIÓN
# (doble torneo) y no dentro de la aptitud: si el tamaño pesara igual que el
# error desde el principio, la población colapsaría en circuitos diminutos con
# 1 ó 2 errores (convergencia prematura) - efecto que se observó al probarlo.
if not hasattr(creator, "AptitudCircuito"):
    creator.create("AptitudCircuito", base.Fitness, weights=(1.0,))
    creator.create("ArbolCircuito", gp.PrimitiveTree, fitness=creator.AptitudCircuito)


def mutacion_mixta(individuo, expr, pset):
    r = random.random()
    if r < 0.5:
        return gp.mutUniform(individuo, expr=expr, pset=pset)       # subárbol
    if r < 0.8:
        return gp.mutNodeReplacement(individuo, pset=pset)          # puntual
    return gp.mutShrink(individuo)                                   # encoger


def seleccion_lexicografica(poblacion, k, tournsize):
    return [min(random.sample(poblacion, tournsize), key=clave) for _ in range(k)]


def construir_toolbox(pset, objetivo, cuidado, args):
    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individuo", tools.initIterate, creator.ArbolCircuito, tb.expr)
    tb.register("poblacion", tools.initRepeat, list, tb.individuo)
    tb.register("evaluate", evaluar, pset=pset, objetivo=objetivo, cuidado=cuidado)
    # Doble torneo (Luke & Panait): torneo por aptitud y luego, entre los
    # ganadores, un torneo "suave" por tamaño (parsimonia).
    tb.register("select_busqueda", tools.selDoubleTournament, fitness_size=args.torneo,
                parsimony_size=1.2, fitness_first=True)
    # En la fase de refinamiento (ya existe un circuito perfecto) el torneo es
    # lexicográfico: (errores, compuertas, nodos) -> empuja a circuitos mínimos.
    tb.register("select_refinamiento", seleccion_lexicografica, tournsize=args.torneo)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=3)
    tb.register("mutate", mutacion_mixta, expr=tb.expr_mut, pset=pset)
    limite = gp.staticLimit(key=operator.attrgetter("height"), max_value=args.altura_max)
    tb.decorate("mate", limite)
    tb.decorate("mutate", limite)
    return tb


def variar(poblacion, tb, pc, pm):
    """Reproducción + cruce + mutación (equivalente a algorithms.varAnd)."""
    hijos = [tb.clone(ind) for ind in poblacion]
    for i in range(1, len(hijos), 2):
        if random.random() < pc:
            hijos[i - 1], hijos[i] = tb.mate(hijos[i - 1], hijos[i])
            del hijos[i - 1].fitness.values, hijos[i].fitness.values
    for i in range(len(hijos)):
        if random.random() < pm:
            (hijos[i],) = tb.mutate(hijos[i])
            del hijos[i].fitness.values
    return hijos


def evolucionar_segmento(seg, pset, args, semilla, verbose=True):
    """Una corrida completa de PG (sección 4.3) para un segmento."""
    random.seed(semilla)
    objetivo = objetivo_segmento(seg)
    cuidado = mascara_cuidado(args.modo)
    tb = construir_toolbox(pset, objetivo, cuidado, args)

    poblacion = tb.poblacion(n=args.poblacion)                    # generación 0
    for ind in poblacion:
        ind.fitness.values = tb.evaluate(ind)

    historia = {"gen": [], "mejor_fapt": [], "media_fapt": [], "mejor_compuertas": [],
                "media_tamano": []}
    gen_perfecto, mejor = None, None
    for gen in range(args.generaciones + 1):
        # Designación del resultado (paso 6): mejor individuo hasta ahora
        elite = sorted(poblacion, key=clave)[:args.elitismo]
        if mejor is None or clave(elite[0]) < clave(mejor):
            mejor = tb.clone(elite[0])
        fap = np.array([ind.fitness.values[0] for ind in poblacion])
        historia["gen"].append(gen)
        historia["mejor_fapt"].append(mejor.fitness.values[0])
        historia["media_fapt"].append(float(fap.mean()))
        historia["mejor_compuertas"].append(n_compuertas(mejor))
        historia["media_tamano"].append(float(np.mean([len(i) for i in poblacion])))

        if gen_perfecto is None and errores_de(mejor) == 0:
            gen_perfecto = gen
        if gen_perfecto is not None and gen - gen_perfecto >= args.refinamiento:
            break
        if gen == args.generaciones:
            break

        # Selección basada en la aptitud (torneo) + operadores genéticos.
        # Tras hallar un circuito perfecto se aumenta la presión de parsimonia.
        seleccionar = tb.select_busqueda if gen_perfecto is None else tb.select_refinamiento
        hijos = seleccionar(poblacion, len(poblacion) - args.elitismo)
        hijos = variar(hijos, tb, args.pc, args.pm)
        for ind in hijos:
            if not ind.fitness.valid:
                ind.fitness.values = tb.evaluate(ind)
        # Elitismo: los mejores pasan intactos a la nueva generación
        poblacion = hijos + [tb.clone(e) for e in elite]

    if verbose:
        estado = f"perfecto en gen {gen_perfecto}" if gen_perfecto is not None else "NO perfecto"
        print(f"   segmento {seg}: errores={errores_de(mejor)}  compuertas={n_compuertas(mejor):2d}"
              f"  nodos={len(mejor):2d}  altura={mejor.height}  ({estado}, semilla {semilla})")
    return mejor, historia, gen_perfecto


# =============================================================================
# POST-PROCESAMIENTO: árboles explícitos, recorridos, simplificación, netlist
# =============================================================================
class Nodo:
    """Nodo de árbol n-ario (enlace a hijos + información), como la fig. 4.8."""

    __slots__ = ("nombre", "hijos")

    def __init__(self, nombre, hijos=()):
        self.nombre = nombre
        self.hijos = list(hijos)

    def es_hoja(self):
        return not self.hijos


def deap_a_nodo(individuo):
    """Convierte la lista prefija de DEAP en un árbol enlazado."""
    pila = []
    for elem in reversed(individuo):
        if isinstance(elem, gp.Primitive):
            hijos = [pila.pop() for _ in range(elem.arity)]
            pila.append(Nodo(elem.name, hijos))
        else:
            pila.append(Nodo(str(elem.value)))   # 'A', 'B', 'C' o 'D'
    return pila[0]


def preorden(nodo):
    """Raíz, subárboles izquierdo..derecho (sección 4.5)."""
    res = [nodo.nombre]
    for h in nodo.hijos:
        res += preorden(h)
    return res


def posorden(nodo):
    """Subárboles y luego la raíz -> notación polaca inversa (sección 4.8)."""
    res = []
    for h in nodo.hijos:
        res += posorden(h)
    return res + [nodo.nombre]


def cuenta_nodos(nodo):
    """Algoritmo Cuenta_Nodos (sección 4.5)."""
    return 1 + sum(cuenta_nodos(h) for h in nodo.hijos)


def ejecutar_polaca(expresion_pos, valores):
    """Ejecuta la expresión en posorden con una pila (sección 4.8)."""
    pila = []
    for simbolo in expresion_pos:
        if simbolo in valores:
            pila.append(valores[simbolo])
        elif simbolo == "NOT":
            pila.append(NOT(pila.pop()))
        else:
            y, x = pila.pop(), pila.pop()
            pila.append(FUNC_POR_NOMBRE[simbolo](x, y))
    assert len(pila) == 1
    return pila[0]


def compuertas_nodo(nodo):
    """Número de compuertas (nodos internos) de un árbol enlazado."""
    return 0 if nodo.es_hoja() else 1 + sum(compuertas_nodo(h) for h in nodo.hijos)


def evaluar_nodo(nodo, valores=None):
    valores = valores or VALORES_HOJAS
    if nodo.es_hoja():
        return valores[nodo.nombre]
    return FUNC_POR_NOMBRE[nodo.nombre](*[evaluar_nodo(h, valores) for h in nodo.hijos])


def canonica(nodo):
    """Forma canónica (hijos ordenados en compuertas conmutativas)."""
    if nodo.es_hoja():
        return nodo.nombre
    hijos = [canonica(h) for h in nodo.hijos]
    if nodo.nombre in CONMUTATIVAS:
        hijos.sort()
    return f"{nodo.nombre}({','.join(hijos)})"


def _simplificar_paso(nodo):
    if nodo.es_hoja():
        return nodo
    hijos = [_simplificar_paso(h) for h in nodo.hijos]
    n = nodo.nombre
    cero, uno = Nodo("0"), Nodo("1")
    if n == "NOT":
        x = hijos[0]
        if x.nombre == "NOT":
            return x.hijos[0]                               # (x')' = x
        if x.nombre in ("0", "1"):
            return uno if x.nombre == "0" else cero
        return Nodo(n, hijos)
    x, y = hijos
    cx, cy = canonica(x), canonica(y)
    complementos = (cx == f"NOT({cy})") or (cy == f"NOT({cx})")
    consts = {h.nombre for h in hijos} & {"0", "1"}
    otro = lambda c: y if x.nombre == c else x
    if n == "AND":
        if "0" in consts or complementos:
            return cero                                     # x·0 = 0 ; x·x' = 0
        if "1" in consts:
            return otro("1")                                # x·1 = x
        if cx == cy:
            return x                                        # x·x = x
    if n == "OR":
        if "1" in consts or complementos:
            return uno                                      # x+1 = 1 ; x+x' = 1
        if "0" in consts:
            return otro("0")                                # x+0 = x
        if cx == cy:
            return x                                        # x+x = x
    if n == "XOR":
        if cx == cy:
            return cero                                     # x⊕x = 0
        if complementos:
            return uno                                      # x⊕x' = 1
        if "0" in consts:
            return otro("0")                                # x⊕0 = x
        if "1" in consts:
            return Nodo("NOT", [otro("1")])                 # x⊕1 = x'
    return Nodo(n, hijos)


def simplificar(nodo):
    """
    Post-procesamiento algebraico (álgebra de Boole) hasta punto fijo.
    Elimina redundancias que la evolución deja, p. ej. (D ⊕ D) ⊕ y  ->  y.
    """
    while True:
        nuevo = _simplificar_paso(nodo)
        if canonica(nuevo) == canonica(nodo):
            return nuevo
        nodo = nuevo


def a_infija(nodo, estilo="texto"):
    """Expresión infija: 'texto' (·,+,⊕,'), 'verilog' o 'python'."""
    ops = {
        "texto": {"AND": "·", "OR": " + ", "XOR": " ⊕ "},
        "verilog": {"AND": " & ", "OR": " | ", "XOR": " ^ "},
        "python": {"AND": " & ", "OR": " | ", "XOR": " ^ "},
    }[estilo]
    if nodo.es_hoja():
        if estilo == "verilog" and nodo.nombre in ("0", "1"):
            return "1'b" + nodo.nombre
        return nodo.nombre
    n = nodo.nombre
    hs = [a_infija(h, estilo) for h in nodo.hijos]
    if n == "NOT":
        if estilo == "texto":
            return f"{hs[0]}'" if nodo.hijos[0].es_hoja() else f"({hs[0]})'"
        return f"(~{hs[0]} & 1)" if estilo == "python" else f"~{hs[0]}"
    if n in ("NAND", "NOR"):
        base_op = {"NAND": "AND", "NOR": "OR"}[n]
        interno = f"({hs[0]}{ops[base_op]}{hs[1]})"
        if estilo == "texto":
            return f"{interno}'"
        return f"(~{interno} & 1)" if estilo == "python" else f"~{interno}"
    return f"({hs[0]}{ops[n]}{hs[1]})"


def compuertas_compartidas(arboles):
    """
    Netlist con 'structural hashing': subexpresiones idénticas en distintos
    segmentos se implementan UNA sola vez en el hardware.
    """
    unicas = {}

    def visitar(nodo):
        if nodo.es_hoja():
            return
        unicas[canonica(nodo)] = nodo.nombre
        for h in nodo.hijos:
            visitar(h)

    for arbol in arboles.values():
        visitar(arbol)
    return unicas


# =============================================================================
# DISEÑO CLÁSICO DE REFERENCIA: Quine-McCluskey (equivalente a Karnaugh)
# =============================================================================
def quine_mccluskey(minterms, dontcares, n_vars=4):
    """Devuelve una suma de productos mínima (implicantes como cadenas '1-0-')."""
    def combinar(a, b):
        dif = [i for i in range(n_vars) if a[i] != b[i]]
        if len(dif) == 1 and "-" not in (a[dif[0]], b[dif[0]]):
            i = dif[0]
            return a[:i] + "-" + a[i + 1:]
        return None

    termino = lambda m: format(m, f"0{n_vars}b")
    actuales = {termino(m) for m in minterms | dontcares}
    primos = set()
    while actuales:
        usados, siguientes = set(), set()
        lista = sorted(actuales)
        for i in range(len(lista)):
            for j in range(i + 1, len(lista)):
                c = combinar(lista[i], lista[j])
                if c:
                    siguientes.add(c)
                    usados.update((lista[i], lista[j]))
        primos |= actuales - usados
        actuales = siguientes

    cubre = lambda imp, m: all(c == "-" or c == b for c, b in zip(imp, termino(m)))
    por_cubrir, seleccion = set(minterms), []
    # Implicantes primos esenciales
    for m in sorted(minterms):
        candidatos = [p for p in primos if cubre(p, m)]
        if len(candidatos) == 1 and candidatos[0] not in seleccion:
            seleccion.append(candidatos[0])
    for p in seleccion:
        por_cubrir -= {m for m in por_cubrir if cubre(p, m)}
    # Cobertura voraz del resto (prefiere implicantes grandes)
    while por_cubrir:
        p = max(primos, key=lambda p: (sum(cubre(p, m) for m in por_cubrir), p.count("-")))
        seleccion.append(p)
        por_cubrir -= {m for m in por_cubrir if cubre(p, m)}
    return seleccion


def costo_sop(implicantes):
    """Compuertas de 2 entradas equivalentes + inversores de una SOP."""
    literales = [[(v, c) for v, c in zip("ABCD", imp) if c != "-"] for imp in implicantes]
    ands = sum(max(len(l) - 1, 0) for l in literales)
    ors = max(len(implicantes) - 1, 0)
    nots = len({v for l in literales for v, c in l if c == "0"})
    return ands + ors + nots


def sop_texto(implicantes):
    terminos = []
    for imp in implicantes:
        lit = [v + ("'" if c == "0" else "") for v, c in zip("ABCD", imp) if c != "-"]
        terminos.append("·".join(lit) if lit else "1")
    return " + ".join(terminos)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
def display_ascii(bits):
    """bits: dict segmento->0/1.  Devuelve 3 líneas del dígito."""
    s = lambda k, ch: ch if bits[k] else " "
    return [" " + s("a", "_") + " ",
            s("f", "|") + s("g", "_") + s("b", "|"),
            s("e", "|") + s("d", "_") + s("c", "|")]


def salida_circuito(arboles, n):
    return {seg: (evaluar_nodo(arboles[seg]) >> n) & 1 for seg in SEGMENTOS}


def graficar(historias, arboles, carpeta, modo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # ---- Convergencia (análoga a la figura 4.12 del documento) --------------
    fig, ejes = plt.subplots(2, 4, figsize=(16, 7))
    for k, seg in enumerate(SEGMENTOS):
        ax = ejes.flat[k]
        h = historias[seg]
        ax.plot(h["gen"], h["mejor_fapt"], color="tab:blue", lw=2, label="mejor f_apt")
        ax.plot(h["gen"], h["media_fapt"], color="tab:blue", lw=1, ls="--", label="media f_apt")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 15)
        ax.set_title(f"Segmento {seg}")
        ax.set_xlabel("generación")
        ax.set_ylabel("f_apt (log)")
        ax2 = ax.twinx()
        ax2.plot(h["gen"], h["mejor_compuertas"], color="tab:red", lw=1.5, label="compuertas del mejor")
        ax2.set_ylabel("compuertas", color="tab:red")
        if k == 0:
            l1, e1 = ax.get_legend_handles_labels()
            l2, e2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, e1 + e2, fontsize=7, loc="lower right")
    ejes.flat[7].axis("off")
    ejes.flat[7].text(0.0, 0.5,
                      "f_apt = 1 / (0.1 + Σ|y − yc|)\nmáximo = 10  (circuito perfecto)\n\n"
                      "Tras hallar la solución, la presión\nde parsimonia sigue reduciendo\n"
                      "el número de compuertas.", fontsize=11, va="center")
    fig.suptitle("Historia de la aptitud por segmento – PG para decodificador BCD→7 segmentos")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "convergencia_segmentos.png"), dpi=130)
    plt.close(fig)

    # ---- Árboles-programa ----------------------------------------------------
    def posiciones(nodo, prof=0, cont=[0], pos=None, aristas=None):
        if nodo.es_hoja():
            x = cont[0]
            cont[0] += 1
        else:
            xs = [posiciones(h, prof + 1, cont, pos, aristas)[0] for h in nodo.hijos]
            x = sum(xs) / len(xs)
        pos[id(nodo)] = (x, -prof, nodo)
        for h in nodo.hijos:
            aristas.append((id(nodo), id(h)))
        return x, pos, aristas

    fig, ejes = plt.subplots(2, 4, figsize=(20, 10))
    for k, seg in enumerate(SEGMENTOS):
        ax = ejes.flat[k]
        pos, aristas = {}, []
        posiciones(arboles[seg], 0, [0], pos, aristas)
        for a, b in aristas:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="gray", lw=1, zorder=1)
        for x, y, nodo in pos.values():
            color = "#a8d0f0" if nodo.es_hoja() else "#f6c28b"
            ax.scatter([x], [y], s=650, color=color, edgecolor="k", zorder=2)
            ax.text(x, y, nodo.nombre, ha="center", va="center", fontsize=7, zorder=3)
        ax.set_title(f"Segmento {seg}  ({len(pos) - sum(n.es_hoja() for *_, n in pos.values())} compuertas)")
        ax.axis("off")
        ax.margins(0.15)
    ejes.flat[7].axis("off")
    ejes.flat[7].text(0.05, 0.5, "Naranja: funciones (compuertas)\nAzul: terminales (A, B, C, D)\n"
                      "A = bit más significativo", fontsize=12, va="center")
    fig.suptitle("Árboles-programa evolucionados (uno por segmento)")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "arboles_segmentos.png"), dpi=110)
    plt.close(fig)

    # ---- Dígitos dibujados por el circuito ----------------------------------
    geometria = {  # polígonos de cada segmento en un dígito de 1 x 2
        "a": [(0.15, 2.0), (0.85, 2.0), (0.75, 1.9), (0.25, 1.9)],
        "b": [(0.9, 1.95), (0.9, 1.05), (0.8, 1.1), (0.8, 1.85)],
        "c": [(0.9, 0.95), (0.9, 0.05), (0.8, 0.15), (0.8, 0.9)],
        "d": [(0.15, 0.0), (0.85, 0.0), (0.75, 0.1), (0.25, 0.1)],
        "e": [(0.1, 0.95), (0.1, 0.05), (0.2, 0.15), (0.2, 0.9)],
        "f": [(0.1, 1.95), (0.1, 1.05), (0.2, 1.1), (0.2, 1.85)],
        "g": [(0.2, 1.0), (0.3, 1.06), (0.7, 1.06), (0.8, 1.0), (0.7, 0.94), (0.3, 0.94)],
    }
    filas = list(range(16)) if modo == "completo" else list(range(10))
    fig, ejes = plt.subplots(1, len(filas), figsize=(1.3 * len(filas), 3))
    for ax, n in zip(ejes, filas):
        bits = salida_circuito(arboles, n)
        for seg, pts in geometria.items():
            ax.add_patch(Polygon(pts, closed=True, color="red" if bits[seg] else "#eeeeee"))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 2.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{n:04b}", fontsize=9)
    fig.suptitle("Salida del circuito evolucionado para cada entrada BCD")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "display_evolucionado.png"), dpi=130)
    plt.close(fig)


# =============================================================================
# EXPORTACIÓN (interfaz de salida, como MEPX genera código - sección 4.10)
# =============================================================================
def exportar_verilog(arboles, ruta, conjunto):
    lineas = [
        "// Decodificador BCD -> 7 segmentos (cátodo común)",
        f"// Diseñado automáticamente por Programación Genética (DEAP), conjunto '{conjunto}'",
        "module decodificador_7seg_pg (",
        "    input  wire A, B, C, D,          // A = MSB",
        "    output wire seg_a, seg_b, seg_c, seg_d, seg_e, seg_f, seg_g",
        ");",
    ]
    for seg in SEGMENTOS:
        lineas.append(f"    assign seg_{seg} = {a_infija(arboles[seg], 'verilog')};")
    lineas.append("endmodule")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


def exportar_python(arboles, ruta):
    cuerpo = "\n".join(f"    seg_{seg} = {a_infija(arboles[seg], 'python')}" for seg in SEGMENTOS)
    codigo = f'''# -*- coding: utf-8 -*-
"""Decodificador BCD -> 7 segmentos generado por Programación Genética."""


def decodificar(A, B, C, D):
    """Recibe los 4 bits BCD (A = MSB) y devuelve los segmentos (a..g) como 0/1."""
{cuerpo}
    return seg_a, seg_b, seg_c, seg_d, seg_e, seg_f, seg_g


def dibujar(digito):
    a, b, c, d, e, f, g = decodificar(*[(digito >> k) & 1 for k in (3, 2, 1, 0)])
    return [" " + ("_" if a else " ") + " ",
            ("|" if f else " ") + ("_" if g else " ") + ("|" if b else " "),
            ("|" if e else " ") + ("_" if d else " ") + ("|" if c else " ")]


if __name__ == "__main__":
    lineas = [dibujar(n) for n in range(10)]
    for fila in range(3):
        print("  ".join(l[fila] for l in lineas))
'''
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(codigo)


# =============================================================================
# EXPERIMENTO OPCIONAL: ¿qué conjunto de funciones conviene?
# =============================================================================
def comparar_conjuntos(args, n_semillas):
    """
    Mide, para cada conjunto de funciones F, la probabilidad de éxito de UNA
    corrida (sin reintentos), las generaciones hasta la solución y el tamaño
    de los circuitos correctos. Ayuda a justificar la elección del paso 3.
    """
    print("\n" + "=" * 78)
    print(f" EXPERIMENTO: comparación de conjuntos de funciones ({n_semillas} semillas x 7 segmentos)")
    print("=" * 78)
    print(f" {'conjunto':<10} {'corridas exitosas':>18} {'gen. media a solución':>22} "
          f"{'compuertas/segmento':>20}")
    for conjunto in ("basico", "extendido", "nand"):
        pset = construir_pset(conjunto)
        exitos, gens, tamanos, total = 0, [], [], 0
        for s in range(n_semillas):
            for k, seg in enumerate(SEGMENTOS):
                mejor, _, gp_ok = evolucionar_segmento(seg, pset, args, 1000 * s + k, verbose=False)
                total += 1
                if gp_ok is not None:
                    exitos += 1
                    gens.append(gp_ok)
                    tamanos.append(n_compuertas(mejor))
        tasa = f"{exitos}/{total} ({100 * exitos / total:.0f}%)"
        print(f" {conjunto:<10} {tasa:>18} {np.mean(gens) if gens else float('nan'):>22.1f}"
              f" {np.mean(tamanos) if tamanos else float('nan'):>20.1f}")
    print(" (compuertas/segmento: promedio solo sobre las corridas que hallaron un circuito correcto)")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
class Tee:
    """Escribe simultáneamente en consola y en el archivo de reporte."""

    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, texto):
        for f in self.flujos:
            f.write(texto)

    def flush(self):
        for f in self.flujos:
            f.flush()


def main():
    p = argparse.ArgumentParser(description="PG para diseñar un decodificador BCD a 7 segmentos")
    p.add_argument("--compuertas", choices=list(CONJUNTOS_FUNCIONES), default="extendido")
    p.add_argument("--modo", choices=["dontcare", "completo"], default="dontcare")
    p.add_argument("--poblacion", type=int, default=300)
    p.add_argument("--generaciones", type=int, default=200)
    p.add_argument("--refinamiento", type=int, default=80,
                   help="generaciones extra tras hallar un circuito perfecto")
    p.add_argument("--pc", type=float, default=0.7, help="probabilidad de cruce")
    p.add_argument("--pm", type=float, default=0.2, help="probabilidad de mutación")
    p.add_argument("--torneo", type=int, default=4)
    p.add_argument("--elitismo", type=int, default=2)
    p.add_argument("--altura-max", dest="altura_max", type=int, default=8)
    p.add_argument("--reintentos", type=int, default=3,
                   help="corridas adicionales si un segmento no llega a error 0")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="resultados_ej1")
    p.add_argument("--sin-graficas", action="store_true")
    p.add_argument("--comparar-conjuntos", type=int, default=0, metavar="N",
                   help="ejecuta además el experimento de conjuntos de funciones con N semillas")
    args = p.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    reporte = open(os.path.join(args.salida, "reporte_ej1.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, reporte)
    t0 = time.time()

    print("=" * 78)
    print(" PG - DISEÑO DEL DECODIFICADOR BCD -> 7 SEGMENTOS")
    print("=" * 78)
    pset = construir_pset(args.compuertas)
    print(" Terminales T :", list(ENTRADAS))
    print(" Funciones  F :", [f"{n}/{a}" for n, _, a in CONJUNTOS_FUNCIONES[args.compuertas]],
          f"(conjunto '{args.compuertas}')")
    print(" Aptitud      : f_apt = 1/(0.1 + Σ|y_j − yc_j|)  [máx 10]  + mín. compuertas")
    print(f" Modo         : {args.modo}  ->  filas evaluadas: "
          f"{contar_unos(mascara_cuidado(args.modo))} de 16")
    print(f" Parámetros   : pobl={args.poblacion}, gen={args.generaciones}, pc={args.pc}, "
          f"pm={args.pm}, torneo={args.torneo}, elitismo={args.elitismo}, altura≤{args.altura_max}")
    print("\n Tabla de verdad objetivo:")
    print("   dig  A B C D   a b c d e f g")
    for n, patron in TABLA_7SEG.items():
        print(f"    {n}   {' '.join(format(n, '04b'))}   {' '.join(patron)}")
    if args.modo == "completo":
        print("   10-15         0 0 0 0 0 0 0   (display apagado)")
    else:
        print("   10-15         x x x x x x x   (no importa / don't care)")

    # ---------------------- Evolución de los 7 segmentos ---------------------
    print("\n EVOLUCIÓN")
    mejores, historias = {}, {}
    for k, seg in enumerate(SEGMENTOS):
        mejor, hist = None, None
        for intento in range(args.reintentos + 1):
            cand, h, _ = evolucionar_segmento(seg, pset, args, args.semilla + 97 * k + 7919 * intento)
            if mejor is None or clave(cand) < clave(mejor):
                mejor, hist = cand, h
            if errores_de(mejor) == 0:
                break
        mejores[seg], historias[seg] = mejor, hist

    # ---------------------- Post-procesamiento --------------------------------
    arboles = {seg: simplificar(deap_a_nodo(ind)) for seg, ind in mejores.items()}
    cuidado = mascara_cuidado(args.modo)

    print("\n RESULTADOS POR SEGMENTO")
    print(" " + "-" * 76)
    for seg in SEGMENTOS:
        arbol = arboles[seg]
        pos = posorden(arbol)
        salida_pila = ejecutar_polaca(pos, VALORES_HOJAS)
        assert salida_pila == evaluar_nodo(arbol), "la ejecución polaca no coincide"
        print(f" Segmento {seg}   f_apt = {mejores[seg].fitness.values[0]:.4f}   "
              f"nodos (Cuenta_Nodos): evolucionado = {len(mejores[seg])}, "
              f"simplificado = {cuenta_nodos(arbol)}")
        print(f"   Programa evolucionado  : {mejores[seg]}")
        print(f"   Preorden               : {' '.join(preorden(arbol))}")
        print(f"   Posorden (polaca inv.) : {' '.join(pos)}")
        print(f"   Expresión booleana     : {seg} = {a_infija(arbol)}")

    # ---------------------- Verificación exhaustiva ---------------------------
    print("\n VERIFICACIÓN EXHAUSTIVA DE LA TABLA DE VERDAD")
    errores_totales = 0
    for seg in SEGMENTOS:
        diff = (evaluar_nodo(arboles[seg]) ^ objetivo_segmento(seg)) & cuidado
        errores_totales += contar_unos(diff)
    filas = range(16) if args.modo == "completo" else range(10)
    lineas = [display_ascii(salida_circuito(arboles, n)) for n in filas]
    for r in range(3):
        print("   " + "  ".join(l[r] for l in lineas))
    print("   " + "  ".join(f"{n:>3}" if n > 9 else f" {n} " for n in filas))
    print(f"   Errores totales: {errores_totales}  ->  "
          f"{'CIRCUITO CORRECTO' if errores_totales == 0 else 'circuito con errores'}")
    if args.modo == "dontcare":
        print("   (Salida del circuito para 10..15, no especificada por el problema:)")
        extra = [display_ascii(salida_circuito(arboles, n)) for n in range(10, 16)]
        for r in range(3):
            print("   " + "  ".join(l[r] for l in extra))

    # ---------------------- Costo del hardware y comparación ------------------
    print("\n COSTO DEL CIRCUITO Y COMPARACIÓN CON DISEÑO CLÁSICO (Quine-McCluskey / Karnaugh)")
    print(f"   {'seg':<4}{'PG: compuertas':>15}{'QM: compuertas*':>17}   SOP mínima (QM)")
    dontcares = set(range(10, 16)) if args.modo == "dontcare" else set()
    total_pg = total_qm = 0
    for seg in SEGMENTOS:
        minterms = {n for n, pat in TABLA_7SEG.items() if pat[SEGMENTOS.index(seg)] == "1"}
        imp = quine_mccluskey(minterms, dontcares)
        c_pg = compuertas_nodo(arboles[seg])
        c_qm = costo_sop(imp)
        total_pg += c_pg
        total_qm += c_qm
        print(f"   {seg:<4}{c_pg:>15}{c_qm:>17}   {seg} = {sop_texto(imp)}")
    compartidas = compuertas_compartidas(arboles)
    print(f"   {'Σ':<4}{total_pg:>15}{total_qm:>17}")
    print(f"   Compuertas del circuito PG compartiendo subexpresiones entre segmentos: "
          f"{len(compartidas)}")
    print("   * QM: compuertas de 2 entradas equivalentes + inversores, sin compartir.")

    # ---------------------- Salidas -------------------------------------------
    exportar_verilog(arboles, os.path.join(args.salida, "decodificador_7seg_pg.v"), args.compuertas)
    exportar_python(arboles, os.path.join(args.salida, "decodificador_7seg_pg.py"))
    if not args.sin_graficas:
        graficar(historias, arboles, args.salida, args.modo)

    if args.comparar_conjuntos:
        comparar_conjuntos(args, args.comparar_conjuntos)

    print(f"\n Archivos generados en {os.path.abspath(args.salida)}   (tiempo total {time.time() - t0:.1f} s)")
    sys.stdout = sys.__stdout__
    reporte.close()


if __name__ == "__main__":
    main()
