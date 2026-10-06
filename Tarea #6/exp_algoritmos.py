"""
Ejercicios 2, 3 y 4: SVM, K-NN y Árboles de decisión aplicados a datos del
Instituto Nacional de Salud (datos.gov.co) para estimar el riesgo de fallecimiento
por COVID-19. Todas las implementaciones son propias (desde cero) y se contrastan
con scikit-learn solo como verificación.
"""
import json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from datos import particion, metricas_binarias, vpp_prevalencia, SEMILLA
from svm_desde_cero import SVM
from knn_desde_cero import KNN
from arbol_desde_cero import ArbolDecision
from estilo import plt, FIG, AZUL, NARANJA, AQUA, TINTA, TINTA2, GRILLA

Xtr_df, Xte_df, ytr, yte = particion()
NOMBRES = list(Xtr_df.columns)
Xtr, Xte = Xtr_df.values.astype(float), Xte_df.values.astype(float)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
PLIEGUES = list(skf.split(Xtr, ytr))
PREV_REAL = 0.02          # letalidad ilustrativa (orden de magnitud observado en Colombia)
R = {"datos": dict(n_ent=int(len(ytr)), n_prueba=int(len(yte)), atributos=NOMBRES,
                   pos_ent=int(ytr.sum()), pos_prueba=int(yte.sum()))}

def cv(fabrica, X=Xtr, y=ytr, prep=None):
    """Exactitud media y desviación en validación cruzada de 5 pliegues."""
    acc = []
    for a, b in PLIEGUES:
        Xa, Xb = X[a], X[b]
        if prep:
            Xa, Xb = prep(Xa, Xb)
        acc.append((fabrica().fit(Xa, y[a]).predict(Xb) == y[b]).mean())
    return float(np.mean(acc)), float(np.std(acc))

def estandarizar(Xa, Xb):
    mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
    return (Xa - mu) / sd, (Xb - mu) / sd

# ============================================================== línea base "tradicional"
# Regla escrita a mano (programación tradicional, figura 6.1): "si edad >= u, fallecido".
mejor_u, mejor_acc = None, 0
for u in range(20, 91):
    acc = ((Xtr[:, 0] >= u).astype(int) == ytr).mean()
    if acc > mejor_acc:
        mejor_u, mejor_acc = u, acc
base = metricas_binarias(yte, (Xte[:, 0] >= mejor_u).astype(int))
R["base"] = dict(umbral=mejor_u, acc_ent=float(mejor_acc), prueba=base)
print("Regla base edad >=", mejor_u, base)

# ============================================================== EJERCICIO 2 · SVM
Xtr_s, Xte_s = estandarizar(Xtr, Xte)
grid_lin = [0.01, 0.1, 1, 10, 100]
res_lin = {C: cv(lambda C=C: SVM(C=C, kernel="lineal"), Xtr_s) for C in grid_lin}
Cs, gammas = [0.1, 1, 10, 100], [0.01, 0.03, 0.1, 0.3, 1.0]
M = np.zeros((len(Cs), len(gammas)))
for i, C in enumerate(Cs):
    for j, g in enumerate(gammas):
        M[i, j] = cv(lambda C=C, g=g: SVM(C=C, kernel="rbf", gamma=g), Xtr_s)[0]
i, j = np.unravel_index(np.argmax(M), M.shape)
C_best, g_best = Cs[i], gammas[j]
best_lin = max(res_lin, key=lambda c: res_lin[c][0])
print("SVM lineal CV", res_lin, "RBF mejor", C_best, g_best, M.max())

t0 = time.perf_counter()
svm = SVM(C=C_best, kernel="rbf", gamma=g_best).fit(Xtr_s, ytr)
t_svm = time.perf_counter() - t0
pred_svm = svm.predict(Xte_s)
m_svm = metricas_binarias(yte, pred_svm)
sk = SVC(C=C_best, kernel="rbf", gamma=g_best).fit(Xtr_s, ytr)
acuerdo_svm = float((sk.predict(Xte_s) == pred_svm).mean())
svm_lin = SVM(C=best_lin, kernel="lineal").fit(Xtr_s, ytr)
m_svm_lin = metricas_binarias(yte, svm_lin.predict(Xte_s))
w_lin = (svm_lin.coef_dual_ @ svm_lin.vectores_soporte_)          # w = sum a_i y_i x_i
cv_svm = cv(lambda: SVM(C=C_best, kernel="rbf", gamma=g_best), Xtr_s)

# Cambio 1: costo asimétrico
pesos = {}
for w in [1, 1.5, 2, 3, 4]:
    m = SVM(C=C_best, kernel="rbf", gamma=g_best, peso_clase={1: w}).fit(Xtr_s, ytr)
    pesos[w] = metricas_binarias(yte, m.predict(Xte_s))

# Cambios 2 y 3: Platt con valores de decisión fuera de pliegue + corrección por prevalencia
f_oof = np.zeros(len(ytr))
for a, b in PLIEGUES:
    Xa, Xb = estandarizar(Xtr[a], Xtr[b])
    f_oof[b] = SVM(C=C_best, kernel="rbf", gamma=g_best).fit(Xa, ytr[a]).decision_function(Xb)
svm.calibrar(None, ytr, f=f_oof)
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
def perfil(edad, sexoM, rel=0, asint=0, demora=8, dep="Bogota"):
    x = np.zeros(len(NOMBRES)); x[0] = edad; x[1] = sexoM; x[2] = rel; x[3] = asint; x[4] = demora
    x[NOMBRES.index(f"dep_{dep}")] = 1
    return (x - mu) / sd
perfiles = [(25, 0, "Bogota"), (45, 1, "Antioquia"), (60, 0, "Valle"), (70, 1, "Bogota"), (85, 0, "Atlantico")]
Xp = np.array([perfil(e, s, dep=d) for e, s, d in perfiles])
p_bal = svm.predict_proba(Xp); p_real = svm.predict_proba(Xp, prevalencia=PREV_REAL)
tabla_perfiles = [dict(edad=e, sexo="M" if s else "F", dep=d, f=float(fv), p_bal=float(pb), p_real=float(pr))
                  for (e, s, d), fv, pb, pr in zip(perfiles, svm.decision_function(Xp), p_bal, p_real)]
print("perfiles", tabla_perfiles, "Platt", svm.platt_)

R["svm"] = dict(cv_lineal={str(k): v for k, v in res_lin.items()}, grid_C=Cs, grid_g=gammas, M=M.tolist(),
                C=C_best, gamma=g_best, cv=cv_svm, prueba=m_svm, n_sv=svm.n_sv_, n_iter=svm.n_iter_,
                t_fit=t_svm, acuerdo_sklearn=acuerdo_svm, lineal=dict(C=best_lin, prueba=m_svm_lin,
                w={n: float(v) for n, v in zip(NOMBRES, w_lin)}, b=float(svm_lin.b_)),
                pesos={str(k): v for k, v in pesos.items()}, platt=list(svm.platt_), perfiles=tabla_perfiles,
                vpp_real=vpp_prevalencia(m_svm["sensibilidad"], m_svm["especificidad"], PREV_REAL))

# Figura: mapa de calor de la validación cruzada
fig, ax = plt.subplots(figsize=(5.2, 3.0))
im = ax.imshow(M, cmap="Blues", vmin=M.min() - 0.02, vmax=M.max(), aspect="auto")
ax.set_xticks(range(len(gammas)), [str(g).replace(".", ",") for g in gammas]); ax.set_yticks(range(len(Cs)), [str(c).replace(".", ",") for c in Cs])
ax.set_xlabel("γ (ancho del núcleo RBF)"); ax.set_ylabel("C")
ax.grid(False)
for a in range(len(Cs)):
    for b in range(len(gammas)):
        ax.text(b, a, f"{M[a, b]*100:.1f}".replace(".", ","), ha="center", va="center", fontsize=8,
                color="white" if M[a, b] > (M.min() + M.max()) / 2 else TINTA, fontweight="bold" if (a, b) == (i, j) else None)
ax.set_title("Exactitud en validación cruzada (%) - SVM RBF")
fig.colorbar(im, ax=ax, fraction=0.04).outline.set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/ej2_cv_heatmap.png"); plt.close(fig)

# Figura: frontera de decisión en 2D (edad, demora) para ver margen y vectores de soporte
X2 = Xtr[:, [0, 4]]; mu2, sd2 = X2.mean(0), X2.std(0)
m2 = SVM(C=1, kernel="rbf", gamma=0.5).fit((X2 - mu2) / sd2, ytr)
gx, gy = np.meshgrid(np.linspace(0, 100, 300), np.linspace(0, 40, 200))
Z = m2.decision_function((np.c_[gx.ravel(), gy.ravel()] - mu2) / sd2).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.contourf(gx, gy, Z, levels=[-50, 0, 50], colors=["#dbe8f7", "#fbe0d4"], alpha=0.9)
ax.contour(gx, gy, Z, levels=[-1, 0, 1], colors=[TINTA2, TINTA, TINTA2], linestyles=["--", "-", "--"], linewidths=[1, 1.6, 1])
jit = np.random.default_rng(0).uniform(-0.35, 0.35, size=len(ytr))
for c, col, lab in [(0, AZUL, "Recuperado"), (1, NARANJA, "Fallecido")]:
    s = ytr == c
    ax.scatter(X2[s, 0], X2[s, 1] + jit[s], s=10, color=col, label=lab, alpha=0.75, linewidths=0)
sv = (m2.vectores_soporte_ * sd2 + mu2)
ax.scatter(sv[:, 0], sv[:, 1], s=30, facecolors="none", edgecolors=TINTA, linewidths=0.5, label="Vectores de soporte")
ax.set_xlim(0, 100); ax.set_ylim(0, 40)
ax.set_xlabel("Edad (años)"); ax.set_ylabel("Días síntomas → diagnóstico")
ax.set_title("Frontera SVM-RBF (línea continua) y márgenes f(x) = ±1")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5, ncol=3)
ax.grid(False)
fig.tight_layout(); fig.savefig(f"{FIG}/ej2_frontera.png"); plt.close(fig)
R["svm"]["frontera_2d"] = dict(n_sv=m2.n_sv_, acc_prueba=float((m2.predict((Xte[:, [0, 4]] - mu2) / sd2) == yte).mean()))

# ============================================================== EJERCICIO 3 · K-NN
# Matriz alternativa para HEOM: departamento como UNA columna categórica (0..3)
dep_cols = [n for n in NOMBRES if n.startswith("dep_")]
def a_heom(Xdf):
    Z = Xdf.drop(columns=dep_cols).copy()
    Z["departamento"] = np.argmax(Xdf[dep_cols].values, 1)
    return Z
Htr, Hte = a_heom(Xtr_df).values.astype(float), a_heom(Xte_df).values.astype(float)
cat_heom = [1, 2, 3, 5]          # sexo_M, fuente_relacionado, asintomatico, departamento
ks = list(range(1, 52, 2))
variantes = {
    "Euclídea sin estandarizar": (lambda k: KNN(k=k, estandarizar=False), Xtr, Xte),
    "Euclídea estandarizada": (lambda k: KNN(k=k), Xtr, Xte),
    "Estandarizada + voto ponderado": (lambda k: KNN(k=k, pesos="distancia"), Xtr, Xte),
    "HEOM + voto ponderado": (lambda k: KNN(k=k, pesos="distancia", metrica="heom", categoricas=cat_heom), Htr, Hte),
}
curvas, knn_res = {}, {}
for nom, (fab, A, B) in variantes.items():
    cvs = [cv(lambda k=k: fab(k), A) for k in ks]
    curvas[nom] = [c[0] for c in cvs]
    kb = ks[int(np.argmax(curvas[nom]))]
    mod = fab(kb).fit(A, ytr)
    knn_res[nom] = dict(k=kb, cv=cvs[ks.index(kb)], prueba=metricas_binarias(yte, mod.predict(B)))
    print("KNN", nom, knn_res[nom])
fab_f, A_f, B_f = variantes["HEOM + voto ponderado"]
acc_ent_k = [(KNN(k=k, metrica="heom", categoricas=cat_heom).fit(A_f, ytr).predict(A_f) == ytr).mean() for k in ks]
cv_unif_heom = [cv(lambda k=k: KNN(k=k, metrica="heom", categoricas=cat_heom), A_f)[0] for k in ks]
k_final = knn_res["HEOM + voto ponderado"]["k"]
knn_final = KNN(k=k_final, pesos="distancia", metrica="heom", categoricas=cat_heom).fit(Htr, ytr)
# umbral: con probabilidades de vecinos se puede priorizar sensibilidad
umbrales = {}
for u in [0.3, 0.4, 0.5, 0.6]:
    umbrales[u] = metricas_binarias(yte, knn_final.predict(Hte, umbral=u))
# verificación con scikit-learn (variante estandarizada)
kb = knn_res["Euclídea estandarizada"]["k"]
sa, sb = estandarizar(Xtr, Xte)
acuerdo_knn = float((KNeighborsClassifier(kb).fit(sa, ytr).predict(sb) == KNN(k=kb).fit(Xtr, ytr).predict(Xte)).mean())
t0 = time.perf_counter(); knn_final.predict(Hte); t_knn = time.perf_counter() - t0
# vecinos de un paciente ejemplo
idx, d = knn_final.vecinos(Hte[:1])
R["knn"] = dict(ks=ks, curvas=curvas, res=knn_res, acc_ent=list(map(float, acc_ent_k)), cv_unif_heom=cv_unif_heom,
                k_final=k_final, umbrales={str(k): v for k, v in umbrales.items()}, acuerdo_sklearn=acuerdo_knn,
                t_pred=t_knn, vpp_real=vpp_prevalencia(knn_res["HEOM + voto ponderado"]["prueba"]["sensibilidad"],
                                                       knn_res["HEOM + voto ponderado"]["prueba"]["especificidad"], PREV_REAL),
                ejemplo=dict(x=Hte[0].tolist(), y=int(yte[0]), vecinos=Htr[idx[0]].tolist(), yv=ytr[idx[0]].tolist(), d=d[0].tolist(),
                             p=float(knn_final.predict_proba(Hte[:1])[0, 1])))

fig, ax = plt.subplots(figsize=(7.6, 3.3))
cols = {"Euclídea sin estandarizar": "#9a9893", "Euclídea estandarizada": AQUA,
        "Estandarizada + voto ponderado": NARANJA, "HEOM + voto ponderado": AZUL}
for nom, c in curvas.items():
    ax.plot(ks, np.array(c) * 100, color=cols[nom], lw=1.8, label=nom)
ax.plot(ks, np.array(acc_ent_k) * 100, color=AZUL, lw=1.2, ls=":", label="HEOM (voto uniforme) - entrenamiento")
ax.set_xlabel("k (número de vecinos)"); ax.set_ylabel("Exactitud (%)")
ax.set_title("K-NN: exactitud en validación cruzada según k y variante")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(); fig.savefig(f"{FIG}/ej3_k.png"); plt.close(fig)

# ============================================================== EJERCICIO 4 · Árboles de decisión
prof = list(range(1, 13))
acc_tr_p, acc_cv_p, acc_cv_e = [], [], []
for p in prof:
    acc_tr_p.append(float((ArbolDecision(max_prof=p).fit(Xtr, ytr).predict(Xtr) == ytr).mean()))
    acc_cv_p.append(cv(lambda p=p: ArbolDecision(max_prof=p))[0])
    acc_cv_e.append(cv(lambda p=p: ArbolDecision(criterio="entropia", max_prof=p))[0])
completo = ArbolDecision(min_muestras_hoja=1).fit(Xtr, ytr, NOMBRES)
m_comp = metricas_binarias(yte, completo.predict(Xte))
acc_tr_comp = float((completo.predict(Xtr) == ytr).mean())
alphas, hojas = completo.ruta_poda()
# alpha por validación cruzada (se evalúan medias geométricas entre alphas consecutivos)
cands = np.unique(np.r_[0, np.sqrt(alphas[1:-1] * alphas[2:]), alphas[1:]])
cands = cands[cands < alphas[-1]]
cv_alpha = [cv(lambda a=a: ArbolDecision(ccp_alpha=a)) for a in cands]
media = np.array([c[0] for c in cv_alpha]); desv = np.array([c[1] for c in cv_alpha])
k_best = int(np.argmax(media))
# regla de "una desviación estándar": árbol más simple dentro de 1 SE del mejor
se = desv[k_best] / np.sqrt(5)
k_1se = int(np.max(np.where(media >= media[k_best] - se)[0]))
alpha_final = float(cands[k_best])
alpha_1se = float(cands[k_1se])
hojas_1se = ArbolDecision(ccp_alpha=alpha_1se).fit(Xtr, ytr).n_hojas()
podado = ArbolDecision(ccp_alpha=alpha_final).fit(Xtr, ytr, NOMBRES)
m_pod = metricas_binarias(yte, podado.predict(Xte))
cv_pod = cv(lambda: ArbolDecision(ccp_alpha=alpha_final))
sk_t = DecisionTreeClassifier(ccp_alpha=alpha_final, random_state=0).fit(Xtr, ytr)
acuerdo_arbol = float((sk_t.predict(Xte) == podado.predict(Xte)).mean())
pesos_t = {}
for w in [1, 1.5, 2, 3]:
    pesos_t[w] = metricas_binarias(yte, ArbolDecision(ccp_alpha=alpha_final, peso_clase={1: w}).fit(Xtr, ytr).predict(Xte))
fmt = {n: "bin" for n in NOMBRES if n not in ("edad", "demora_dx")}
reglas = podado.reglas({0: "Recuperado", 1: "Fallecido"}, fmt)
R["arbol"] = dict(prof=prof, acc_tr=acc_tr_p, acc_cv_gini=acc_cv_p, acc_cv_ent=acc_cv_e,
                  completo=dict(hojas=completo.n_hojas(), prof=completo.profundidad(), acc_ent=acc_tr_comp, prueba=m_comp),
                  alphas=alphas.tolist(), hojas_ruta=hojas.tolist(), cands=cands.tolist(), cv_alpha=[list(c) for c in cv_alpha],
                  alpha_mejor=float(cands[k_best]), alpha_final=alpha_final, alpha_1se=alpha_1se, hojas_1se=hojas_1se,
                  podado=dict(hojas=podado.n_hojas(), prof=podado.profundidad(), prueba=m_pod, cv=cv_pod,
                              acc_ent=float((podado.predict(Xtr) == ytr).mean())),
                  acuerdo_sklearn=acuerdo_arbol, pesos={str(k): v for k, v in pesos_t.items()}, reglas=reglas,
                  importancia={n: float(v) for n, v in zip(NOMBRES, podado.importancia_)},
                  vpp_real=vpp_prevalencia(m_pod["sensibilidad"], m_pod["especificidad"], PREV_REAL))
print("Árbol completo", R["arbol"]["completo"]); print("Podado", R["arbol"]["podado"], alpha_final)
for r in reglas: print(r)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(prof, np.array(acc_tr_p) * 100, color=NARANJA, marker="o", ms=4, label="Entrenamiento (Gini)")
ax.plot(prof, np.array(acc_cv_p) * 100, color=AZUL, marker="o", ms=4, label="Validación cruzada (Gini)")
ax.plot(prof, np.array(acc_cv_e) * 100, color=AQUA, marker="s", ms=4, label="Validación cruzada (Entropía)")
ax.set_xlabel("Profundidad máxima del árbol"); ax.set_ylabel("Exactitud (%)")
ax.set_title("Árbol de decisión: sobreajuste al crecer la profundidad")
ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{FIG}/ej4_profundidad.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.0))
nh = [ArbolDecision(ccp_alpha=a).fit(Xtr, ytr).n_hojas() for a in cands]
ax.errorbar(nh, media * 100, yerr=desv * 100, color=AZUL, marker="o", ms=4, capsize=2, lw=1.5, elinewidth=0.8)
ax.axvline(podado.n_hojas(), color=NARANJA, ls="--", lw=1.2)
ax.text(podado.n_hojas() + 0.6, media.min() * 100 + 0.5, f"árbol elegido\n({podado.n_hojas()} hojas)", color=TINTA2, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Número de hojas del árbol podado (log)"); ax.set_ylabel("Exactitud CV (%)")
ax.set_title("Poda por costo-complejidad: exactitud CV ± 1 desviación")
fig.tight_layout(); fig.savefig(f"{FIG}/ej4_poda.png"); plt.close(fig)

# dibujo del árbol podado
def dibujar_arbol(arbol, ruta):
    posiciones, aristas = {}, []
    cont = [0]
    def rec(t, prof):
        if t.es_hoja:
            x = cont[0]; cont[0] += 1
        else:
            xi = rec(t.izq, prof + 1); xd = rec(t.der, prof + 1)
            x = (xi + xd) / 2
            aristas.append((id(t), id(t.izq), "sí")); aristas.append((id(t), id(t.der), "no"))
        posiciones[id(t)] = (x, -prof, t)
        return x
    rec(arbol.raiz_, 0)
    nh, pr = cont[0], arbol.profundidad()
    fig, ax = plt.subplots(figsize=(max(6, nh * 1.6), 1.05 * (pr + 1)))
    for a, b, lab in aristas:
        xa, ya, _ = posiciones[a]; xb, yb, _ = posiciones[b]
        ax.plot([xa, xb], [ya, yb], color=GRILLA, lw=1.2, zorder=1)
        ax.text((xa + xb) / 2, (ya + yb) / 2, lab, fontsize=6.5, color=TINTA2, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    for x, y, t in posiciones.values():
        p = t.valor / t.valor.sum()
        if t.es_hoja:
            txt = f"{'Fallecido' if p[1] > 0.5 else 'Recuperado'}\nP(fallece) = " + f"{p[1]:.2f}".replace(".", ",") + f"\nn = {t.n}"
            fc = "#fbe0d4" if p[1] > 0.5 else "#dbe8f7"
        else:
            nom = arbol.nombres_[t.atributo]
            cond = f"{nom} = 0" if nom not in ("edad", "demora_dx") else f"{nom} ≤ {t.umbral:.1f}".replace(".", ",")
            txt = f"{cond}\nGini={t.impureza:.2f}  n={t.n}".replace(".", ",")
            fc = "white"
        ax.text(x, y, txt, ha="center", va="center", fontsize=7, color=TINTA, zorder=2,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=TINTA2, lw=0.6))
    ax.set_xlim(-0.8, nh - 0.2); ax.set_ylim(-pr - 0.6, 0.6); ax.axis("off")
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
dibujar_arbol(podado, f"{FIG}/ej4_arbol.png")

imp = sorted(R["arbol"]["importancia"].items(), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.barh([k for k, v in imp], [v * 100 for k, v in imp], color=AZUL, height=0.6)
for yv, (k, v) in enumerate(imp):
    ax.text(v * 100 + 0.8, yv, f"{v*100:.1f} %".replace(".", ","), va="center", fontsize=7.5, color=TINTA2)
ax.set_xlabel("Importancia (% de la reducción total de impureza)"); ax.grid(axis="y", visible=False)
ax.set_title("Importancia de variables - árbol podado")
fig.tight_layout(); fig.savefig(f"{FIG}/ej4_importancia.png"); plt.close(fig)

# ============================================================== Comparación final
final = {
    f"Regla manual (edad ≥ {mejor_u})": base,
    "SVM RBF": m_svm,
    f"K-NN HEOM (k={k_final})": knn_res["HEOM + voto ponderado"]["prueba"],
    "Árbol podado": m_pod,
}
R["final"] = final
fig, ax = plt.subplots(figsize=(6.8, 3.2))
metr = [("exactitud", "Exactitud", AZUL), ("sensibilidad", "Sensibilidad", NARANJA), ("especificidad", "Especificidad", AQUA)]
xs = np.arange(len(final)); ancho = 0.26
for q, (clave, lab, col) in enumerate(metr):
    vals = [v[clave] * 100 for v in final.values()]
    ax.bar(xs + (q - 1) * ancho, vals, width=ancho - 0.03, color=col, label=lab)
ax.set_xticks(xs, list(final.keys()), fontsize=8); ax.set_ylim(50, 100)
ax.set_ylabel("% en el conjunto de prueba"); ax.grid(axis="x", visible=False)
ax.set_title("Comparación en el conjunto de prueba (n = 200)")
ax.legend(ncol=3, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIG}/comparacion_final.png"); plt.close(fig)

json.dump(R, open("res_algoritmos.json", "w"), indent=1, ensure_ascii=False, default=float)
print("OK")
