"""Verificación de las implementaciones propias contra scikit-learn en data sets de referencia."""
import json, numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from svm_desde_cero import SVM
from knn_desde_cero import KNN
from arbol_desde_cero import ArbolDecision
import pandas as pd

V = {}
X, y = load_breast_cancer(return_X_y=True); Xs = StandardScaler().fit_transform(X)
filas = []
for ker, g in [("lineal", 0), ("rbf", 0.05)]:
    for C in [0.1, 1, 10]:
        m = SVM(C=C, kernel=ker, gamma=g).fit(Xs, y)
        s = SVC(C=C, kernel="linear" if ker == "lineal" else "rbf", gamma=g if g else "scale").fit(Xs, y)
        d1, d2 = m.decision_function(Xs), s.decision_function(Xs)
        filas.append(dict(kernel=ker, C=C, sv_propio=m.n_sv_, sv_sk=int(s.n_support_.sum()), b_propio=m.b_,
                          b_sk=float(s.intercept_[0]), dif_max=float(np.abs(d1 - d2).max()),
                          acuerdo=float((np.sign(d1) == np.sign(d2)).mean())))
V["svm"] = filas

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); ac = []
for k in [1, 3, 5, 11]:
    for p in [1, 2]:
        for w in ["uniforme", "distancia"]:
            a = KNN(k=k, p=p, pesos=w).fit(Xtr, ytr).predict(Xte)
            b = KNeighborsClassifier(k, p=p, weights="uniform" if w == "uniforme" else "distance").fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
            ac.append(dict(k=k, p=p, pesos=w, acuerdo=float((a == b).mean()), acc=float((a == yte).mean())))
V["knn"] = ac

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
at = []
for crit in ["gini", "entropy"]:
    for d in [2, 4, None]:
        m = ArbolDecision(criterio="gini" if crit == "gini" else "entropia", max_prof=d).fit(Xtr, ytr)
        s = DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=0).fit(Xtr, ytr)
        at.append(dict(criterio=crit, prof=d, hojas=m.n_hojas(), hojas_sk=int(s.get_n_leaves()),
                       acc=float((m.predict(Xte) == yte).mean()), acc_sk=float((s.predict(Xte) == yte).mean()),
                       acuerdo=float((m.predict(Xte) == s.predict(Xte)).mean())))
m = ArbolDecision().fit(Xtr, ytr); al, h = m.ruta_poda()
p = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).cost_complexity_pruning_path(Xtr, ytr)
V["arbol"] = at
V["ruta"] = dict(propio=al.tolist(), sk=p.ccp_alphas.tolist(), hojas=h.tolist(),
                 dif_max=float(np.abs(al - p.ccp_alphas).max()))

# Estadística descriptiva del data set gubernamental (análisis exploratorio)
df = pd.read_csv("datos/covid_ins_muestra.csv")
eda = {}
for c, g in df.groupby("fallecido"):
    eda[int(c)] = dict(n=len(g), edad_media=float(g.edad.mean()), edad_mediana=float(g.edad.median()),
                       pct_M=float((g.sexo.str.upper() == "M").mean()), pct_rel=float((g.fuente == "R").mean()),
                       pct_asint=float(g.asintomatico.mean()), demora_media=float(g.demora_dx.mean()),
                       pct_demora15=float((g.demora_dx == 15).mean()),
                       pct_2021p=float((pd.to_datetime(g.fdx) >= "2021-02-01").mean()))
V["eda"] = eda
V["faltantes"] = dict(fis=int(df.fis.isna().sum()), fdx=int(df.fdx.isna().sum()))
json.dump(V, open("res_verificacion.json", "w"), indent=1, default=float)
print(json.dumps(V["eda"], indent=1)); print(V["ruta"]["dif_max"], V["faltantes"])
