"""
Máquina de Vectores de Soporte (SVM) implementada desde cero.

Problema primal (margen suave, con costo por clase):
    min_{w,b,xi}  1/2 ||w||^2 + sum_i C_i xi_i
    s.a.          y_i (w·phi(x_i) + b) >= 1 - xi_i ,   xi_i >= 0,     y_i en {-1, +1}

Problema dual (el que se resuelve):
    min_a  1/2 a^T Q a - e^T a ,   Q_ij = y_i y_j K(x_i, x_j)
    s.a.   0 <= a_i <= C_i ,  y^T a = 0

Solucionador: SMO (Sequential Minimal Optimization) con selección del "par de máxima
violación" de las condiciones KKT (Keerthi et al. 2001; Fan, Chen y Lin 2005, base de
LIBSVM). En cada iteración se optimizan analíticamente solo dos multiplicadores.

Función de decisión:  f(x) = sum_i a_i y_i K(x_i, x) + b ;  clase = signo(f(x)).

CAMBIOS introducidos para la aplicación (triaje de riesgo de COVID-19):
  1. Costo asimétrico por clase (C_i = C * peso[y_i]): penaliza más los falsos negativos.
  2. Escalado de Platt: convierte f(x) en una probabilidad P(y=1|x) = 1/(1+exp(A f + B)).
  3. Corrección por prevalencia: la muestra está balanceada (50 %), pero en la población
     la letalidad es mucho menor; se reajustan las probabilidades con el teorema de Bayes.
"""
import numpy as np


# --------------------------------------------------------------------------- núcleos
def kernel_lineal(X1, X2, **_):
    return X1 @ X2.T

def kernel_rbf(X1, X2, gamma=0.1, **_):
    """K(x, z) = exp(-gamma ||x - z||^2)"""
    d2 = (X1**2).sum(1)[:, None] + (X2**2).sum(1)[None, :] - 2 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(d2, 0.0))

KERNELS = {"lineal": kernel_lineal, "rbf": kernel_rbf}


class SVM:
    """SVM binaria. Etiquetas de entrada en {0,1} o {-1,+1}.

    Parámetros
    ----------
    C : float           costo de violar el margen (C grande -> margen duro, riesgo de sobreajuste)
    kernel : str        'lineal' o 'rbf'
    gamma : float       ancho del núcleo RBF (gamma grande -> fronteras más irregulares)
    peso_clase : dict   {etiqueta: peso}; multiplica C para esa clase (cambio 1)
    tol : float         tolerancia KKT para detener SMO
    max_iter : int      máximo de iteraciones SMO
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.1, peso_clase=None, tol=1e-3, max_iter=100_000):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.peso_clase, self.tol, self.max_iter = peso_clase, tol, max_iter

    # ------------------------------------------------------------------ entrenamiento
    def _K(self, A, B):
        return KERNELS[self.kernel](A, B, gamma=self.gamma)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.clases_ = np.unique(y)
        ys = np.where(y == self.clases_[1], 1.0, -1.0)          # a {-1,+1}
        n = len(ys)
        Cv = np.full(n, float(self.C))
        if self.peso_clase:
            for etiqueta, w in self.peso_clase.items():
                Cv[y == etiqueta] *= w
        K = self._K(X, X)
        Q = (ys[:, None] * ys[None, :]) * K
        a = np.zeros(n)
        G = -np.ones(n)                      # gradiente de la función dual: Q a - e
        it = 0
        while it < self.max_iter:
            # --- selección del par de máxima violación (WSS1)
            menos_yG = -ys * G
            arriba = ((ys > 0) & (a < Cv)) | ((ys < 0) & (a > 0))     # I_up
            abajo = ((ys < 0) & (a < Cv)) | ((ys > 0) & (a > 0))      # I_low
            if not arriba.any() or not abajo.any():
                break
            i = np.where(arriba)[0][np.argmax(menos_yG[arriba])]
            j = np.where(abajo)[0][np.argmin(menos_yG[abajo])]
            if menos_yG[i] - menos_yG[j] < self.tol:                  # KKT satisfechas
                break
            # --- actualización analítica de (a_i, a_j)
            ai_old, aj_old = a[i], a[j]
            Ci, Cj = Cv[i], Cv[j]
            quad = K[i, i] + K[j, j] - 2 * K[i, j]
            quad = quad if quad > 1e-12 else 1e-12
            if ys[i] != ys[j]:
                delta = (-G[i] - G[j]) / quad
                diff = a[i] - a[j]
                a[i] += delta; a[j] += delta
                if diff > 0:
                    if a[j] < 0: a[j] = 0; a[i] = diff
                else:
                    if a[i] < 0: a[i] = 0; a[j] = -diff
                if diff > Ci - Cj:
                    if a[i] > Ci: a[i] = Ci; a[j] = Ci - diff
                else:
                    if a[j] > Cj: a[j] = Cj; a[i] = Cj + diff
            else:
                delta = (G[i] - G[j]) / quad
                s = a[i] + a[j]
                a[i] -= delta; a[j] += delta
                if s > Ci:
                    if a[i] > Ci: a[i] = Ci; a[j] = s - Ci
                else:
                    if a[j] < 0: a[j] = 0; a[i] = s
                if s > Cj:
                    if a[j] > Cj: a[j] = Cj; a[i] = s - Cj
                else:
                    if a[i] < 0: a[i] = 0; a[j] = s
            G += Q[:, i] * (a[i] - ai_old) + Q[:, j] * (a[j] - aj_old)
            it += 1
        self.n_iter_ = it
        # --- sesgo b a partir de los vectores de soporte libres (0 < a < C)
        yG = ys * G
        libres = (a > 1e-8) & (a < Cv - 1e-8)
        if libres.any():
            rho = yG[libres].mean()
        else:
            menos_yG = -ys * G
            arriba = ((ys > 0) & (a < Cv)) | ((ys < 0) & (a > 0))
            abajo = ((ys < 0) & (a < Cv)) | ((ys > 0) & (a > 0))
            rho = -(menos_yG[arriba].max() + menos_yG[abajo].min()) / 2
        self.b_ = -rho
        sv = a > 1e-8
        self.vectores_soporte_ = X[sv]
        self.coef_dual_ = (a * ys)[sv]              # a_i y_i
        self.n_sv_ = int(sv.sum())
        self.alpha_ = a
        return self

    # ------------------------------------------------------------------ predicción
    def decision_function(self, X):
        return self._K(np.asarray(X, float), self.vectores_soporte_) @ self.coef_dual_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.clases_[1], self.clases_[0])

    # ------------------------------------------------------------------ cambio 2: Platt
    def calibrar(self, X, y, iter_newton=100, f=None):
        """Ajusta P(y=1|f) = 1/(1+exp(A f + B)) por máxima verosimilitud (Newton-Raphson),
        con los objetivos suavizados de Platt (1999) para evitar sobreajuste.
        Si se pasa `f` (valores de decisión fuera de pliegue, obtenidos por validación
        cruzada) se usan esos en lugar de recalcularlos sobre los mismos datos de ajuste."""
        f = self.decision_function(X) if f is None else np.asarray(f, float)
        y = (np.asarray(y) == self.clases_[1]).astype(float)
        n1, n0 = y.sum(), len(y) - y.sum()
        t = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
        A, B = 0.0, np.log((n0 + 1) / (n1 + 1))
        for _ in range(iter_newton):
            p = 1 / (1 + np.exp(A * f + B))
            g = np.array([((t - p) * f).sum(), (t - p).sum()])            # gradiente de -logL
            w = p * (1 - p)
            H = np.array([[(w * f * f).sum(), (w * f).sum()], [(w * f).sum(), w.sum()]]) + 1e-10 * np.eye(2)
            paso = np.linalg.solve(H, g)
            A, B = A - paso[0], B - paso[1]
            if np.abs(paso).max() < 1e-10:
                break
        self.platt_ = (A, B)
        return self

    def predict_proba(self, X, prevalencia=None, prevalencia_entrenamiento=0.5):
        """P(y=1|x). Con `prevalencia` aplica el cambio 3 (corrección de probabilidades a priori):
            odds_real = odds_modelo * [pi/(1-pi)] / [pi_ent/(1-pi_ent)]"""
        A, B = self.platt_
        p = 1 / (1 + np.exp(A * self.decision_function(X) + B))
        if prevalencia is not None:
            pi, pi0 = prevalencia, prevalencia_entrenamiento
            odds = p / (1 - p) * (pi / (1 - pi)) / (pi0 / (1 - pi0))
            p = odds / (1 + odds)
        return p
