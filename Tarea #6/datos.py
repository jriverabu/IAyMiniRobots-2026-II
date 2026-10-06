"""
Carga y preparación del data set gubernamental usado en los ejercicios 2, 3 y 4.

Fuente: Instituto Nacional de Salud (INS) - "Casos positivos de COVID-19 en Colombia",
portal de Datos Abiertos del Estado colombiano, www.datos.gov.co, recurso gt2j-8ykr
(Ley 1712 de 2014). Muestra balanceada de 800 casos: 100 fallecidos y 100 recuperados
de cada uno de 4 departamentos (Bogotá, Antioquia, Valle, Atlántico).

Tarea: clasificación binaria  y = 1 (fallecido)  vs  y = 0 (recuperado).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RUTA = "datos/covid_ins_muestra.csv"
SEMILLA = 42
DEPTOS = ["ANTIOQUIA", "ATLANTICO", "BOGOTA", "VALLE"]

def cargar_crudo():
    df = pd.read_csv(RUTA)
    return df

def preparar(df, mediana_demora=None):
    """Limpieza + ingeniería de características (secciones 6.7.3 y 6.7.4 del capítulo).

    - edad: se convierte a años según unidad_medida (1 años, 2 meses, 3 días).
    - sexo_M, fuente_relacionado: variables binarias (0/1).
    - asintomatico: 1 si no hay fecha de inicio de síntomas (dato faltante informativo).
    - demora_dx: días entre inicio de síntomas y diagnóstico; los faltantes se asignan
      con la MEDIANA del conjunto de entrenamiento (asignación por mediana, 6.7.3.2.2).
    - departamento: codificación one-hot (4 columnas).
    """
    X = pd.DataFrame()
    X["edad"] = df["edad"].astype(float)
    X["sexo_M"] = (df["sexo"].str.upper() == "M").astype(int)
    X["fuente_relacionado"] = (df["fuente"] == "R").astype(int)
    X["asintomatico"] = df["asintomatico"].astype(int)
    if mediana_demora is None:
        mediana_demora = float(df["demora_dx"].median())
    X["demora_dx"] = df["demora_dx"].fillna(mediana_demora).clip(0, 60)
    for d in DEPTOS:
        X[f"dep_{d.title()}"] = (df["departamento"] == d).astype(int)
    y = df["fallecido"].astype(int).values
    return X, y, mediana_demora

def particion():
    """Partición estratificada 75 % entrenamiento / 25 % prueba.
    La validación se hace con validación cruzada de 5 pliegues DENTRO del entrenamiento."""
    df = cargar_crudo()
    df_tr, df_te = train_test_split(df, test_size=0.25, stratify=df["fallecido"], random_state=SEMILLA)
    Xtr, ytr, med = preparar(df_tr)
    Xte, yte, _ = preparar(df_te, med)
    return Xtr, Xte, ytr, yte

COLUMNAS_NUM = ["edad", "demora_dx"]

def metricas_binarias(y, yp):
    """Matriz de confusión (figura 6.2) y medidas derivadas."""
    y = np.asarray(y); yp = np.asarray(yp)
    VP = int(((y == 1) & (yp == 1)).sum()); VN = int(((y == 0) & (yp == 0)).sum())
    FP = int(((y == 0) & (yp == 1)).sum()); FN = int(((y == 1) & (yp == 0)).sum())
    n = VP + VN + FP + FN
    exact = (VP + VN) / n
    sens = VP / (VP + FN) if VP + FN else 0.0
    esp = VN / (VN + FP) if VN + FP else 0.0
    prec = VP / (VP + FP) if VP + FP else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return dict(VP=VP, VN=VN, FP=FP, FN=FN, exactitud=exact, sensibilidad=sens,
                especificidad=esp, precision=prec, F1=f1)

def vpp_prevalencia(sens, esp, prev):
    """Valor predictivo positivo esperado si la prevalencia real es `prev` (teorema de Bayes)."""
    return sens * prev / (sens * prev + (1 - esp) * (1 - prev))

if __name__ == "__main__":
    Xtr, Xte, ytr, yte = particion()
    print(Xtr.shape, Xte.shape, ytr.mean(), yte.mean())
    print(Xtr.describe().T)
