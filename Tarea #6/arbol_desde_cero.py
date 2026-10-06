"""
Árbol de decisión CART (Classification And Regression Trees) implementado desde cero.

Construcción recursiva ("divide y vencerás"): en cada nodo se busca la pregunta binaria
    ¿x_a <= umbral?
que más reduce la impureza de los hijos, ponderada por su tamaño:

    Ganancia(a, u) = I(padre) - [ n_izq/n * I(izq) + n_der/n * I(der) ]

Medidas de impureza para una distribución de clases p_c:
    Gini:      I = 1 - sum_c p_c^2
    Entropía:  I = - sum_c p_c log2 p_c       (la ganancia es la "ganancia de información" de ID3/C4.5)

Criterios de parada (pre-poda): profundidad máxima, mínimo de muestras para dividir,
mínimo de muestras por hoja, ganancia mínima.

Post-poda por costo-complejidad (Breiman et al. 1984):
    R_alpha(T) = R(T) + alpha * |hojas(T)|
Para cada nodo interno t se calcula el "eslabón más débil"
    g(t) = [R(t) - R(T_t)] / (|hojas(T_t)| - 1)
y se colapsan sucesivamente los nodos de menor g(t). Se obtiene una secuencia de árboles
anidados, uno por cada alpha; el alpha se elige por validación cruzada.
(R se mide con la impureza ponderada, igual que scikit-learn, para poder comparar.)

CAMBIOS introducidos para la aplicación:
  1. Pesos por clase (costo asimétrico): cada ejemplo aporta peso[y] a los conteos.
  2. Poda por costo-complejidad con alpha elegido por validación cruzada.
  3. Exportación de reglas SI-ENTONCES legibles en español (el modelo deja de ser una
     "caja negra", ver sección 6.11 del capítulo) y cálculo de importancia de variables.
"""
import copy
import numpy as np


def _impureza(conteos, criterio):
    tot = conteos.sum()
    if tot <= 0:
        return 0.0
    p = conteos / tot
    if criterio == "gini":
        return 1.0 - (p**2).sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class Nodo:
    __slots__ = ("atributo", "umbral", "izq", "der", "valor", "n", "peso", "impureza", "prof")

    def __init__(self, valor, n, peso, impureza, prof):
        self.atributo = None; self.umbral = None; self.izq = None; self.der = None
        self.valor, self.n, self.peso, self.impureza, self.prof = valor, n, peso, impureza, prof

    @property
    def es_hoja(self):
        return self.izq is None


class ArbolDecision:
    def __init__(self, criterio="gini", max_prof=None, min_muestras_division=2,
                 min_muestras_hoja=1, ganancia_min=0.0, peso_clase=None, ccp_alpha=0.0):
        self.criterio, self.max_prof = criterio, max_prof
        self.min_div, self.min_hoja, self.ganancia_min = min_muestras_division, min_muestras_hoja, ganancia_min
        self.peso_clase, self.ccp_alpha = peso_clase, ccp_alpha

    # ------------------------------------------------------------------ construcción
    def fit(self, X, y, nombres=None):
        X = np.asarray(X, float); y = np.asarray(y)
        self.clases_ = np.unique(y)
        self.nombres_ = list(nombres) if nombres is not None else [f"x{j}" for j in range(X.shape[1])]
        yi = np.searchsorted(self.clases_, y)
        w = np.ones(len(y))
        if self.peso_clase:
            for c, pc in self.peso_clase.items():
                w[y == c] = pc
        self.n_atr_ = X.shape[1]
        self.peso_total_ = w.sum()
        self.importancia_bruta_ = np.zeros(self.n_atr_)
        self.raiz_ = self._crecer(X, yi, w, 0)
        if self.ccp_alpha > 0:
            self.raiz_ = self._podar_hasta(self.raiz_, self.ccp_alpha)
        self._calc_importancia()
        return self

    def _conteos(self, yi, w):
        return np.bincount(yi, weights=w, minlength=len(self.clases_))

    def _crecer(self, X, yi, w, prof):
        cont = self._conteos(yi, w)
        nodo = Nodo(cont, len(yi), w.sum(), _impureza(cont, self.criterio), prof)
        if (self.max_prof is not None and prof >= self.max_prof) or len(yi) < self.min_div \
                or nodo.impureza == 0.0:
            return nodo
        mejor = self._mejor_division(X, yi, w, nodo)
        if mejor is None:
            return nodo
        a, u = mejor
        m = X[:, a] <= u
        nodo.atributo, nodo.umbral = a, u
        nodo.izq = self._crecer(X[m], yi[m], w[m], prof + 1)
        nodo.der = self._crecer(X[~m], yi[~m], w[~m], prof + 1)
        return nodo

    def _mejor_division(self, X, yi, w, nodo):
        n, K = len(yi), len(self.clases_)
        Y1 = np.zeros((n, K)); Y1[np.arange(n), yi] = w         # conteos ponderados one-hot
        mejor_gan, mejor = self.ganancia_min, None
        tot = nodo.valor
        for a in range(X.shape[1]):
            orden = np.argsort(X[:, a], kind="mergesort")
            xs = X[orden, a]
            izq = np.cumsum(Y1[orden], 0)[:-1]                   # conteos a la izquierda de cada corte
            der = tot - izq
            n_izq = np.arange(1, n)
            validos = (xs[1:] > xs[:-1]) & (n_izq >= self.min_hoja) & (n - n_izq >= self.min_hoja)
            if not validos.any():
                continue
            wi, wd = izq.sum(1), der.sum(1)
            if self.criterio == "gini":
                Ii = 1 - ((izq / np.maximum(wi, 1e-300)[:, None])**2).sum(1)
                Id = 1 - ((der / np.maximum(wd, 1e-300)[:, None])**2).sum(1)
            else:
                pi = izq / np.maximum(wi, 1e-300)[:, None]; pd_ = der / np.maximum(wd, 1e-300)[:, None]
                Ii = -(np.where(pi > 0, pi * np.log2(np.where(pi > 0, pi, 1)), 0)).sum(1)
                Id = -(np.where(pd_ > 0, pd_ * np.log2(np.where(pd_ > 0, pd_, 1)), 0)).sum(1)
            gan = nodo.impureza - (wi * Ii + wd * Id) / nodo.peso
            gan[~validos] = -np.inf
            k = int(np.argmax(gan))
            if gan[k] > mejor_gan + 1e-12:
                mejor_gan = gan[k]
                mejor = (a, (xs[k] + xs[k + 1]) / 2.0)
        return mejor

    # ------------------------------------------------------------------ poda costo-complejidad
    def _R(self, nodo):                      # riesgo del nodo como hoja
        return nodo.impureza * nodo.peso / self.peso_total_

    def _subarbol(self, nodo):               # (R(T_t), número de hojas)
        if nodo.es_hoja:
            return self._R(nodo), 1
        ri, hi = self._subarbol(nodo.izq); rd, hd = self._subarbol(nodo.der)
        return ri + rd, hi + hd

    def _eslabon_mas_debil(self, raiz):
        mejor = [np.inf, []]
        def rec(t):
            if t.es_hoja:
                return
            RT, h = self._subarbol(t)
            g = (self._R(t) - RT) / (h - 1)
            if g < mejor[0] - 1e-15:
                mejor[0], mejor[1] = g, [t]
            elif abs(g - mejor[0]) <= 1e-15:
                mejor[1].append(t)
            rec(t.izq); rec(t.der)
        rec(raiz)
        return mejor

    def _podar_hasta(self, raiz, alpha):
        raiz = copy.deepcopy(raiz)
        while not raiz.es_hoja:
            g, nodos = self._eslabon_mas_debil(raiz)
            if g > alpha:
                break
            for t in nodos:
                t.izq = t.der = None; t.atributo = t.umbral = None
        return raiz

    def ruta_poda(self):
        """Secuencia de alphas efectivos y número de hojas de cada árbol anidado."""
        raiz = copy.deepcopy(self.raiz_)
        alphas, hojas = [0.0], [self._subarbol(raiz)[1]]
        while not raiz.es_hoja:
            g, nodos = self._eslabon_mas_debil(raiz)
            for t in nodos:
                t.izq = t.der = None
            alphas.append(max(g, 0.0)); hojas.append(self._subarbol(raiz)[1])
        return np.array(alphas), np.array(hojas)

    # ------------------------------------------------------------------ predicción
    def _hoja(self, x):
        t = self.raiz_
        while not t.es_hoja:
            t = t.izq if x[t.atributo] <= t.umbral else t.der
        return t

    def predict_proba(self, X):
        X = np.asarray(X, float)
        V = np.array([self._hoja(x).valor for x in X])
        return V / V.sum(1, keepdims=True)

    def predict(self, X):
        return self.clases_[np.argmax(self.predict_proba(X), 1)]

    # ------------------------------------------------------------------ interpretación
    def _calc_importancia(self):
        imp = np.zeros(self.n_atr_)
        def rec(t):
            if t.es_hoja:
                return
            imp[t.atributo] += t.peso * t.impureza - t.izq.peso * t.izq.impureza - t.der.peso * t.der.impureza
            rec(t.izq); rec(t.der)
        rec(self.raiz_)
        self.importancia_ = imp / imp.sum() if imp.sum() > 0 else imp

    def profundidad(self):
        def rec(t):
            return 0 if t.es_hoja else 1 + max(rec(t.izq), rec(t.der))
        return rec(self.raiz_)

    def n_hojas(self):
        return self._subarbol(self.raiz_)[1]

    def reglas(self, etiquetas=None, fmt=None):
        """Lista de reglas SI-ENTONCES, una por hoja."""
        etiquetas = etiquetas or {c: str(c) for c in self.clases_}
        fmt = fmt or {}
        salida = []
        def cond(nombre, op, u):
            f = fmt.get(nombre)
            if f == "bin":
                return f"{nombre} = {'0' if op == '<=' else '1'}"
            return f"{nombre} {op} {u:.1f}".replace(".", ",")
        def rec(t, conds):
            if t.es_hoja:
                p = t.valor / t.valor.sum()
                k = int(np.argmax(p))
                salida.append(dict(condiciones=conds, clase=etiquetas[self.clases_[k]],
                                   prob=float(p[k]), n=int(t.n)))
                return
            nom = self.nombres_[t.atributo]
            rec(t.izq, conds + [cond(nom, "<=", t.umbral)])
            rec(t.der, conds + [cond(nom, ">", t.umbral)])
        rec(self.raiz_, [])
        return salida
