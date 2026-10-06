"""
K vecinos más cercanos (K-NN) implementado desde cero.

Idea: no hay fase de entrenamiento propiamente dicha (aprendizaje "perezoso"): se guardan
los ejemplos y, para un punto nuevo x, se buscan los k ejemplos más cercanos según una
distancia d(x, x_i) y se decide por votación.

    y_hat(x) = argmax_c  sum_{i en N_k(x)}  w_i * 1[y_i = c]

    - voto uniforme:           w_i = 1
    - voto ponderado:          w_i = 1 / (d(x, x_i) + eps)          (cambio 2)

Distancias disponibles:
    - Minkowski de orden p:    d(x,z) = ( sum_a |x_a - z_a|^p )^(1/p)   (p=1 Manhattan, p=2 Euclídea)
    - HEOM (Heterogeneous Euclidean-Overlap Metric, Wilson y Martínez 1997)   (cambio 3)
          d(x,z) = sqrt( sum_a d_a(x_a, z_a)^2 )
          d_a = |x_a - z_a| / rango_a      si el atributo a es numérico
          d_a = 0 si x_a = z_a, 1 si no     si el atributo a es categórico

CAMBIOS introducidos para la aplicación (datos mixtos de COVID-19):
  1. Estandarización interna (z-score) de atributos numéricos con parámetros del
     entrenamiento: sin ella la edad (0-100) dominaría a las variables binarias (0-1).
  2. Voto ponderado por el inverso de la distancia (vecinos cercanos pesan más y se
     rompen los empates).
  3. Distancia HEOM para mezclar atributos numéricos y categóricos sin que la
     codificación one-hot duplique el peso de una variable categórica.
  4. Probabilidad estimada = fracción (ponderada) de vecinos positivos -> permite
     mover el umbral de decisión.
"""
import numpy as np


class KNN:
    def __init__(self, k=5, p=2, pesos="uniforme", metrica="minkowski",
                 estandarizar=True, categoricas=None, eps=1e-9):
        self.k, self.p, self.pesos, self.metrica = k, p, pesos, metrica
        self.estandarizar, self.categoricas, self.eps = estandarizar, categoricas or [], eps

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.y_ = np.asarray(y)
        self.clases_ = np.unique(self.y_)
        n_atr = X.shape[1]
        self.es_cat_ = np.zeros(n_atr, bool)
        self.es_cat_[self.categoricas] = True
        if self.metrica == "heom":
            rango = X.max(0) - X.min(0)
            self.escala_ = np.where(self.es_cat_, 1.0, np.where(rango > 0, rango, 1.0))
            self.centro_ = np.zeros(n_atr)
        elif self.estandarizar:
            self.centro_ = X.mean(0)
            s = X.std(0)
            self.escala_ = np.where(s > 0, s, 1.0)
        else:
            self.centro_, self.escala_ = np.zeros(n_atr), np.ones(n_atr)
        self.X_ = (X - self.centro_) / self.escala_
        return self

    def _distancias(self, Xq):
        Xq = (np.asarray(Xq, float) - self.centro_) / self.escala_
        dif = np.abs(Xq[:, None, :] - self.X_[None, :, :])         # (n_consulta, n_ent, n_atr)
        if self.metrica == "heom":
            dif[:, :, self.es_cat_] = (dif[:, :, self.es_cat_] > 0).astype(float)
            return np.sqrt((dif**2).sum(-1))
        if self.p == np.inf:
            return dif.max(-1)
        return (dif**self.p).sum(-1) ** (1.0 / self.p)

    def vecinos(self, Xq):
        D = self._distancias(Xq)
        idx = np.argsort(D, axis=1, kind="stable")[:, :self.k]
        return idx, np.take_along_axis(D, idx, 1)

    def predict_proba(self, Xq):
        idx, d = self.vecinos(Xq)
        w = np.ones_like(d) if self.pesos == "uniforme" else 1.0 / (d + self.eps)
        votos = np.stack([(w * (self.y_[idx] == c)).sum(1) for c in self.clases_], 1)
        return votos / votos.sum(1, keepdims=True)

    def predict(self, Xq, umbral=None):
        P = self.predict_proba(Xq)
        if umbral is not None and len(self.clases_) == 2:
            return np.where(P[:, 1] >= umbral, self.clases_[1], self.clases_[0])
        return self.clases_[np.argmax(P, 1)]
