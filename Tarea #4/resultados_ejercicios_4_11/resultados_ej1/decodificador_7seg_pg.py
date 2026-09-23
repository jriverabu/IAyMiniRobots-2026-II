# -*- coding: utf-8 -*-
"""Decodificador BCD -> 7 segmentos generado por Programación Genética."""


def decodificar(A, B, C, D):
    """Recibe los 4 bits BCD (A = MSB) y devuelve los segmentos (a..g) como 0/1."""
    seg_a = (((D & (~A & 1)) ^ (~B & 1)) | C)
    seg_b = (~(B & (C ^ D)) & 1)
    seg_c = ((B | D) | (~C & 1))
    seg_d = ((D & A) ^ ((C & (~B & 1)) | ((D ^ (~B & 1)) ^ C)))
    seg_e = (~((B & (~C & 1)) | D) & 1)
    seg_f = (~(((~B & 1) & (C | ((~A & 1) & D))) | (C & D)) & 1)
    seg_g = (((C & D) ^ (C ^ A)) | (B ^ C))
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
