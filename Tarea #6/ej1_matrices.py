"""
Ejercicio 1 - Multiplicación de matrices 2x2 aprendida con Machine Learning.

Se construye un data set de 50.000 productos C = A·B, con A y B matrices 2x2 de
enteros aleatorios en [-20, 20]. Se entrenan tres modelos (regresión lineal,
regresión polinomial de grado 2 y red neuronal MLP), se usan con 10 ejemplos
nuevos, se comparan con el cálculo analítico y se estima el costo computacional.
"""
import json, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from estilo import plt, FIG, AZUL, NARANJA, AQUA, TINTA2

SEMILLA = 42
N = 50_000
rng = np.random.default_rng(SEMILLA)

# ------------------------------------------------------------------ 1. Data set
def generar(n, lo=-20, hi=20, rng=rng):
    A = rng.integers(lo, hi + 1, size=(n, 2, 2))
    B = rng.integers(lo, hi + 1, size=(n, 2, 2))
    X = np.hstack([A.reshape(n, 4), B.reshape(n, 4)]).astype(float)   # a11 a12 a21 a22 b11 b12 b21 b22
    Y = (A @ B).reshape(n, 4).astype(float)                             # c11 c12 c21 c22
    return X, Y

X, Y = generar(N)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=SEMILLA)
ESC_Y = 800.0          # |c_ij| <= 2*20*20 = 800 -> salidas en [-1, 1]

def metricas(Yp, Yv=Yte):
    exacto_elem = (np.rint(Yp) == Yv).mean()
    exacto_mat = (np.rint(Yp) == Yv).all(axis=1).mean()
    return dict(MAE=float(mean_absolute_error(Yv, Yp)),
                RMSE=float(np.sqrt(mean_squared_error(Yv, Yp))),
                R2=float(r2_score(Yv, Yp)),
                exacto_elem=float(exacto_elem), exacto_mat=float(exacto_mat))

# ------------------------------------------------------------------ 2. Modelos
def modelo_lineal():
    return LinearRegression()

def modelo_poli():
    return make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())

class MLPMatrices:
    """MLP con entradas estandarizadas y salidas escaladas a [-1, 1]."""
    def __init__(self, capas=(128, 128), semilla=SEMILLA, max_iter=400):
        self.sx = StandardScaler()
        self.red = MLPRegressor(hidden_layer_sizes=capas, activation="relu", solver="adam",
                                learning_rate_init=1e-3, batch_size=256, max_iter=max_iter,
                                early_stopping=True, validation_fraction=0.1,
                                n_iter_no_change=25, tol=1e-7, random_state=semilla)
    def fit(self, X, Y):
        self.red.fit(self.sx.fit_transform(X), Y / ESC_Y); return self
    def predict(self, X):
        return self.red.predict(self.sx.transform(X)) * ESC_Y

res = {}
entrenados = {}
for nombre, fabrica in [("Regresión lineal", modelo_lineal),
                        ("Regresión polinomial (grado 2)", modelo_poli),
                        ("Red neuronal MLP 8-128-128-4", MLPMatrices)]:
    m = fabrica()
    t0 = time.perf_counter(); m.fit(Xtr, Ytr); t_fit = time.perf_counter() - t0
    r = metricas(m.predict(Xte)); r["t_entrenamiento_s"] = t_fit
    if isinstance(m, MLPMatrices):
        r["epocas"] = int(m.red.n_iter_)
    res[nombre] = r; entrenados[nombre] = m
    print(nombre, r)

lin, poli, mlp = entrenados.values()

# Regla aprendida por el modelo polinomial para c11
pf = poli.named_steps["polynomialfeatures"]
lr = poli.named_steps["linearregression"]
nombres_x = ["a11", "a12", "a21", "a22", "b11", "b12", "b21", "b22"]
feat = pf.get_feature_names_out(nombres_x)
reglas = {}
for j, c in enumerate(["c11", "c12", "c21", "c22"]):
    coef = lr.coef_[j]
    idx = np.argsort(-np.abs(coef))[:3]
    reglas[c] = {"terminos": [(feat[i].replace(" ", "·"), round(float(coef[i]), 6)) for i in idx],
                 "max_resto": float(np.sort(np.abs(coef))[-3]),
                 "intercepto": float(lr.intercept_[j])}
print(reglas)

# ------------------------------------------------------------------ 3. Diez ejemplos nuevos
rng10 = np.random.default_rng(2026)
X10, Y10 = generar(10, rng=rng10)
P10_poli = poli.predict(X10); P10_mlp = mlp.predict(X10); P10_lin = lin.predict(X10)
ejemplos = []
for i in range(10):
    ejemplos.append(dict(A=X10[i, :4].astype(int).tolist(), B=X10[i, 4:].astype(int).tolist(),
                         C=Y10[i].astype(int).tolist(),
                         poli=np.round(P10_poli[i], 3).tolist(),
                         mlp=np.round(P10_mlp[i], 2).tolist(),
                         lin=np.round(P10_lin[i], 1).tolist()))
res10 = {k: metricas(P, Y10) for k, P in [("lin", P10_lin), ("poli", P10_poli), ("mlp", P10_mlp)]}

# ------------------------------------------------------------------ 4. Costo computacional
flops = {
    "Analítico (definición)": 8 + 4,                            # 8 mult + 4 sumas
    "Regresión polinomial": 36 + 2 * 44 * 4 + 4,                # 36 productos + 4 salidas x 44 MAC + sesgos
    "MLP 8-128-128-4": 2 * 8 + 2 * (8*128 + 128*128 + 128*4) + (128 + 128 + 4) + 256,  # escalado + MAC + sesgos + ReLU
}

def cronometrar(f, rep=7):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

def mult_python(Xs):
    out = []
    for x in Xs:
        a11, a12, a21, a22, b11, b12, b21, b22 = x
        out.append((a11*b11 + a12*b21, a11*b12 + a12*b22, a21*b11 + a22*b21, a21*b12 + a22*b22))
    return out

def mult_numpy(Xs):
    A = Xs[:, :4].reshape(-1, 2, 2); B = Xs[:, 4:].reshape(-1, 2, 2)
    return A @ B

Xbig, _ = generar(100_000, rng=np.random.default_rng(7))
tiempos = {}
for etiqueta, Xs in [("10 ejemplos", X10), ("100.000 ejemplos", Xbig)]:
    tiempos[etiqueta] = {
        "Analítico (Python puro)": cronometrar(lambda: mult_python(Xs.tolist()), 3 if len(Xs) > 10 else 50),
        "Analítico (NumPy)": cronometrar(lambda: mult_numpy(Xs)),
        "Regresión polinomial": cronometrar(lambda: poli.predict(Xs)),
        "MLP": cronometrar(lambda: mlp.predict(Xs)),
    }
print(tiempos)
epocas = res["Red neuronal MLP 8-128-128-4"]["epocas"]
n_ent = int(len(Xtr) * 0.9)
flops_ent_mlp = 3 * (flops["MLP 8-128-128-4"]) * n_ent * epocas   # fwd + bwd ~ 3x fwd
flops_ent_poli = 2 * len(Xtr) * 44**2 + 44**3                   # ecuaciones normales aprox.

# ------------------------------------------------------------------ 5. Extrapolación (fuera del rango de entrenamiento)
Xext, Yext = generar(5000, lo=-100, hi=100, rng=np.random.default_rng(99))
extrap = {"poli": metricas(poli.predict(Xext), Yext), "mlp": metricas(mlp.predict(Xext), Yext)}
print("extrapolación", extrap)

# ------------------------------------------------------------------ 6. Curva de aprendizaje (tamaño del data set)
tamanos = [500, 1000, 2500, 5000, 10000, 20000, 40000]
curva = {"n": tamanos, "mlp": [], "poli": []}
for n in tamanos:
    m = MLPMatrices(max_iter=400).fit(Xtr[:n], Ytr[:n])
    curva["mlp"].append(metricas(m.predict(Xte))["MAE"])
    p = modelo_poli().fit(Xtr[:n], Ytr[:n])
    curva["poli"].append(max(metricas(p.predict(Xte))["MAE"], 1e-12))
    print(n, curva["mlp"][-1], curva["poli"][-1])

# ------------------------------------------------------------------ Figuras
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True, sharey=True)
sub = slice(0, 1500)
for ax, (nom, m), col in zip(axs, entrenados.items(), [NARANJA, AQUA, AZUL]):
    p = m.predict(Xte[sub])[:, 0]
    ax.scatter(Yte[sub, 0], p, s=6, color=col, alpha=0.55, linewidths=0)
    ax.plot([-800, 800], [-800, 800], color=TINTA2, lw=1, ls="--")
    ax.set_title(nom.replace(" (grado 2)", "\n(grado 2)").replace(" MLP", "\nMLP"), fontsize=9)
    ax.set_xlabel("c11 real")
axs[0].set_ylabel("c11 predicho")
fig.tight_layout(); fig.savefig(f"{FIG}/ej1_pred_vs_real.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(tamanos, curva["mlp"], marker="o", ms=5, color=AZUL, label="MLP 8-128-128-4")
ax.plot(tamanos, curva["poli"], marker="s", ms=5, color=AQUA, label="Regresión polinomial (grado 2)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Ejemplos de entrenamiento"); ax.set_ylabel("MAE en prueba (log)")
ax.set_title("Error de prueba vs. tamaño del data set")
ax.legend(loc="center right")
fig.tight_layout(); fig.savefig(f"{FIG}/ej1_curva_aprendizaje.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 2.8))
metodos = list(tiempos["100.000 ejemplos"].keys())
vals = [tiempos["100.000 ejemplos"][k] * 1e3 for k in metodos]
ax.barh(metodos[::-1], vals[::-1], color=AZUL, height=0.55)
ax.set_xscale("log"); ax.set_xlabel("Tiempo para 100.000 productos (ms, escala log)")
ax.set_title("Costo de inferencia medido")
for y, v in enumerate(vals[::-1]):
    ax.text(v * 1.15, y, f"{v:,.1f} ms".replace(",", "X").replace(".", ",").replace("X", "."), va="center", fontsize=8, color=TINTA2)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/ej1_tiempos.png"); plt.close(fig)

json.dump(dict(res=res, reglas=reglas, ejemplos=ejemplos, res10=res10, flops=flops, tiempos=tiempos,
               flops_ent_mlp=flops_ent_mlp, flops_ent_poli=flops_ent_poli, epocas=epocas,
               extrap=extrap, curva=curva), open("res_ej1.json", "w"), indent=1, ensure_ascii=False)
print("OK")
