# -*- coding: utf-8 -*-
"""
Métricas en el conjunto de PRUEBA:
  AUC-ROC=0.9433  AUC-PR=0.7321  precision=0.709  recall=0.678  F1=0.693
"""
import math

UMBRAL = 3.5010647039574785
_MEDIA = {"M": 4.67841548951619, "I": 6.539517876749689, "D": 0.8957091531970924}
_DESV = {"M": 0.6159038676449315, "I": 0.3216811686024147, "D": 0.563515984577315}


def div(a, b):
    return a / b if abs(b) > 1e-6 else 1.0


def puntaje_riesgo(monto, hora, distancia_km, ingreso_mensual):
    """Puntaje de riesgo: mayor valor = más sospechosa la transacción."""
    M = (math.log10(monto) - _MEDIA["M"]) / _DESV["M"]
    I = (math.log10(ingreso_mensual) - _MEDIA["I"]) / _DESV["I"]
    D = (math.log10(1 + distancia_km) - _MEDIA["D"]) / _DESV["D"]
    HC = math.cos(2 * math.pi * hora / 24)
    HS = math.sin(2 * math.pi * hora / 24)
    return max(HS, (max(HS, M) + (D if M > D else (max(D, HC) if -0.61 > M else M))))


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
        print(f"{desc:<45} puntaje={s:9.3f}  ->  {'POSIBLE FRAUDE' if s >= UMBRAL else 'normal'}")
