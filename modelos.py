"""Modelos matematicos para simulacion por consola.

Incluye:
- Interpolacion de Lagrange.
- Metodo de punto fijo.
- Modelo de crecimiento logistico discreto.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple


@dataclass
class FixedPointStep:
    iteracion: int
    x_anterior: float
    x_actual: float
    error: float


@dataclass
class FixedPointResult:
    convergio: bool
    aproximacion: float
    pasos: List[FixedPointStep]


def interpolacion_lagrange(puntos: Sequence[Tuple[float, float]], x_eval: float) -> float:
    """Evalua el polinomio de Lagrange para x_eval."""
    if len(puntos) < 2:
        raise ValueError("Se necesitan al menos dos puntos para interpolar.")

    xs = [x for x, _ in puntos]
    if len(set(xs)) != len(xs):
        raise ValueError("Los valores de x deben ser distintos.")

    resultado = 0.0
    n = len(puntos)
    for i in range(n):
        xi, yi = puntos[i]
        termino = yi
        for j in range(n):
            if i == j:
                continue
            xj, _ = puntos[j]
            termino *= (x_eval - xj) / (xi - xj)
        resultado += termino
    return resultado


def _evaluar_expresion(expr: str, x: float) -> float:
    """Evalua g(x) de forma controlada usando funciones de math."""
    entorno: Dict[str, float] = {
        "x": x,
        "pi": math.pi,
        "e": math.e,
    }

    # Exponer funciones comunes de math.
    funciones_permitidas = (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "exp",
        "log",
        "log10",
        "sqrt",
        "fabs",
        "sinh",
        "cosh",
        "tanh",
    )
    for nombre in funciones_permitidas:
        entorno[nombre] = getattr(math, nombre)

    try:
        valor = eval(expr, {"__builtins__": {}}, entorno)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"No se pudo evaluar la expresion: {exc}") from exc

    try:
        return float(valor)
    except (TypeError, ValueError) as exc:
        raise ValueError("La expresion no devolvio un numero real.") from exc


def metodo_punto_fijo(
    g_expr: str,
    x0: float,
    tolerancia: float = 1e-6,
    max_iter: int = 100,
) -> FixedPointResult:
    """Ejecuta iteracion de punto fijo x_{n+1} = g(x_n)."""
    if tolerancia <= 0:
        raise ValueError("La tolerancia debe ser mayor a cero.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser mayor a cero.")

    pasos: List[FixedPointStep] = []
    x_anterior = x0

    for iteracion in range(1, max_iter + 1):
        x_actual = _evaluar_expresion(g_expr, x_anterior)
        error = abs(x_actual - x_anterior)
        pasos.append(
            FixedPointStep(
                iteracion=iteracion,
                x_anterior=x_anterior,
                x_actual=x_actual,
                error=error,
            )
        )

        if error < tolerancia:
            return FixedPointResult(convergio=True, aproximacion=x_actual, pasos=pasos)

        x_anterior = x_actual

    return FixedPointResult(convergio=False, aproximacion=x_anterior, pasos=pasos)


def crecimiento_logistico(r: float, k: float, x0: float, pasos: int) -> List[float]:
    """Simula un modelo logistico discreto.

    x_{n+1} = r * x_n * (1 - x_n / K)
    """
    if pasos < 1:
        raise ValueError("El numero de pasos debe ser mayor o igual a 1.")
    if k == 0:
        raise ValueError("K no puede ser cero.")

    valores = [x0]
    x_actual = x0
    for _ in range(pasos):
        x_actual = r * x_actual * (1 - x_actual / k)
        valores.append(x_actual)
    return valores
