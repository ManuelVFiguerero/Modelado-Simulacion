"""Motores numericos para modelado y simulacion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple


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


@dataclass
class BisectionStep:
    iteracion: int
    a: float
    b: float
    c: float
    fc: float
    error_intervalo: float


@dataclass
class BisectionResult:
    convergio: bool
    aproximacion: float
    pasos: List[BisectionStep]


@dataclass
class NewtonStep:
    iteracion: int
    x_anterior: float
    x_actual: float
    error: float
    fx_actual: float


@dataclass
class NewtonResult:
    convergio: bool
    aproximacion: float
    pasos: List[NewtonStep]


@dataclass
class RK4Step:
    paso: int
    t: float
    y: float


@dataclass
class AitkenStep:
    iteracion: int
    xn: float
    xn1: float
    xn2: float
    x_aitken: float
    error: float


@dataclass
class AitkenResult:
    convergio: bool
    aproximacion: float
    pasos: List[AitkenStep]


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


def _entorno_matematico(variables: Mapping[str, float]) -> Dict[str, Any]:
    """Arma un entorno seguro para evaluar expresiones matematicas."""
    entorno: Dict[str, Any] = {
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "pow": pow,
    }
    entorno.update(variables)

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
        "floor",
        "ceil",
    )
    for nombre in funciones_permitidas:
        entorno[nombre] = getattr(math, nombre)
    return entorno


def _evaluar_expresion(expr: str, **variables: float) -> float:
    """Evalua una expresion con variables dadas y funciones de math."""
    entorno = _entorno_matematico(variables)

    try:
        valor = eval(expr, {"__builtins__": {}}, entorno)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"No se pudo evaluar la expresion: {exc}") from exc

    try:
        return float(valor)
    except (TypeError, ValueError) as exc:
        raise ValueError("La expresion no devolvio un numero real.") from exc


def biseccion(
    f_expr: str,
    a: float,
    b: float,
    tolerancia: float = 1e-6,
    max_iter: int = 100,
) -> BisectionResult:
    """Resuelve f(x)=0 por el metodo de biseccion."""
    if tolerancia <= 0:
        raise ValueError("La tolerancia debe ser mayor a cero.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser mayor a cero.")
    if a >= b:
        raise ValueError("El intervalo debe cumplir a < b.")

    fa = _evaluar_expresion(f_expr, x=a)
    fb = _evaluar_expresion(f_expr, x=b)
    if fa == 0:
        return BisectionResult(convergio=True, aproximacion=a, pasos=[])
    if fb == 0:
        return BisectionResult(convergio=True, aproximacion=b, pasos=[])
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en [a, b] (Bolzano no aplica).")

    pasos: List[BisectionStep] = []
    izquierda, derecha = a, b
    f_izquierda = fa

    for iteracion in range(1, max_iter + 1):
        c = (izquierda + derecha) / 2.0
        fc = _evaluar_expresion(f_expr, x=c)
        error_intervalo = abs(derecha - izquierda) / 2.0
        pasos.append(
            BisectionStep(
                iteracion=iteracion,
                a=izquierda,
                b=derecha,
                c=c,
                fc=fc,
                error_intervalo=error_intervalo,
            )
        )

        if abs(fc) < tolerancia or error_intervalo < tolerancia:
            return BisectionResult(convergio=True, aproximacion=c, pasos=pasos)

        if f_izquierda * fc < 0:
            derecha = c
        else:
            izquierda = c
            f_izquierda = fc

    aproximacion = (izquierda + derecha) / 2.0
    return BisectionResult(convergio=False, aproximacion=aproximacion, pasos=pasos)


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
        x_actual = _evaluar_expresion(g_expr, x=x_anterior)
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


def newton_raphson(
    f_expr: str,
    df_expr: str,
    x0: float,
    tolerancia: float = 1e-6,
    max_iter: int = 100,
) -> NewtonResult:
    """Resuelve f(x)=0 por el metodo de Newton-Raphson."""
    if tolerancia <= 0:
        raise ValueError("La tolerancia debe ser mayor a cero.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser mayor a cero.")

    pasos: List[NewtonStep] = []
    x_anterior = x0

    for iteracion in range(1, max_iter + 1):
        fx = _evaluar_expresion(f_expr, x=x_anterior)
        dfx = _evaluar_expresion(df_expr, x=x_anterior)
        if abs(dfx) < 1e-14:
            raise ValueError("Derivada nula o muy cercana a cero.")

        x_actual = x_anterior - fx / dfx
        error = abs(x_actual - x_anterior)
        fx_actual = _evaluar_expresion(f_expr, x=x_actual)
        pasos.append(
            NewtonStep(
                iteracion=iteracion,
                x_anterior=x_anterior,
                x_actual=x_actual,
                error=error,
                fx_actual=fx_actual,
            )
        )

        if error < tolerancia:
            return NewtonResult(convergio=True, aproximacion=x_actual, pasos=pasos)
        x_anterior = x_actual

    return NewtonResult(convergio=False, aproximacion=x_anterior, pasos=pasos)


def diferencia_central(f_expr: str, x: float, h: float = 1e-4) -> float:
    """Aproxima f'(x) usando diferencias centrales."""
    if h <= 0:
        raise ValueError("h debe ser mayor a cero.")
    f_mas = _evaluar_expresion(f_expr, x=x + h)
    f_menos = _evaluar_expresion(f_expr, x=x - h)
    return (f_mas - f_menos) / (2.0 * h)


def aitken_delta_cuadrado(secuencia: Sequence[float]) -> float:
    """Acelera una secuencia usando Delta-Cuadrado de Aitken."""
    if len(secuencia) < 3:
        raise ValueError("Se requieren al menos 3 terminos consecutivos.")

    x0, x1, x2 = secuencia[-3], secuencia[-2], secuencia[-1]
    denominador = x2 - 2.0 * x1 + x0
    if abs(denominador) < 1e-14:
        raise ValueError("No se puede aplicar Aitken: denominador cercano a cero.")

    return x0 - ((x1 - x0) ** 2) / denominador


def aitken_desde_punto_fijo(
    g_expr: str,
    x0: float,
    tolerancia: float = 1e-6,
    max_iter: int = 100,
) -> AitkenResult:
    """Aplica Aitken Delta-Cuadrado sobre una secuencia de punto fijo."""
    if tolerancia <= 0:
        raise ValueError("La tolerancia debe ser mayor a cero.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser mayor a cero.")

    pasos: List[AitkenStep] = []
    xn = x0
    x_aitken_anterior: float | None = None

    for iteracion in range(1, max_iter + 1):
        xn1 = _evaluar_expresion(g_expr, x=xn)
        xn2 = _evaluar_expresion(g_expr, x=xn1)
        denominador = xn2 - 2.0 * xn1 + xn
        if abs(denominador) < 1e-14:
            raise ValueError(
                "No se puede aplicar Aitken: denominador cercano a cero en iteracion "
                f"{iteracion}."
            )

        x_aitken = xn - ((xn1 - xn) ** 2) / denominador
        if x_aitken_anterior is None:
            error = abs(x_aitken - xn)
        else:
            error = abs(x_aitken - x_aitken_anterior)

        pasos.append(
            AitkenStep(
                iteracion=iteracion,
                xn=xn,
                xn1=xn1,
                xn2=xn2,
                x_aitken=x_aitken,
                error=error,
            )
        )

        if x_aitken_anterior is not None and error < tolerancia:
            return AitkenResult(convergio=True, aproximacion=x_aitken, pasos=pasos)

        x_aitken_anterior = x_aitken
        xn = xn1

    aproximacion = x_aitken_anterior if x_aitken_anterior is not None else x0
    return AitkenResult(convergio=False, aproximacion=aproximacion, pasos=pasos)


def runge_kutta_4(
    ode_expr: str,
    t0: float,
    y0: float,
    h: float,
    pasos: int,
) -> List[RK4Step]:
    """Integra y' = f(t, y) con Runge-Kutta de orden 4."""
    if h <= 0:
        raise ValueError("h debe ser mayor a cero.")
    if pasos < 1:
        raise ValueError("pasos debe ser mayor o igual a 1.")

    t = t0
    y = y0
    trayectoria: List[RK4Step] = [RK4Step(paso=0, t=t, y=y)]

    for paso in range(1, pasos + 1):
        k1 = _evaluar_expresion(ode_expr, t=t, y=y)
        k2 = _evaluar_expresion(ode_expr, t=t + h / 2.0, y=y + h * k1 / 2.0)
        k3 = _evaluar_expresion(ode_expr, t=t + h / 2.0, y=y + h * k2 / 2.0)
        k4 = _evaluar_expresion(ode_expr, t=t + h, y=y + h * k3)

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
        trayectoria.append(RK4Step(paso=paso, t=t, y=y))

    return trayectoria


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


class MetodosNumericos:
    """Wrapper orientado a objetos para consumir desde otras apps."""

    @staticmethod
    def biseccion(
        f_expr: str, a: float, b: float, tolerancia: float = 1e-6, max_iter: int = 100
    ) -> BisectionResult:
        return biseccion(f_expr, a, b, tolerancia, max_iter)

    @staticmethod
    def punto_fijo(
        g_expr: str, x0: float, tolerancia: float = 1e-6, max_iter: int = 100
    ) -> FixedPointResult:
        return metodo_punto_fijo(g_expr, x0, tolerancia, max_iter)

    @staticmethod
    def newton_raphson(
        f_expr: str,
        df_expr: str,
        x0: float,
        tolerancia: float = 1e-6,
        max_iter: int = 100,
    ) -> NewtonResult:
        return newton_raphson(f_expr, df_expr, x0, tolerancia, max_iter)

    @staticmethod
    def lagrange(puntos: Sequence[Tuple[float, float]], x_eval: float) -> float:
        return interpolacion_lagrange(puntos, x_eval)

    @staticmethod
    def diferencia_central(f_expr: str, x: float, h: float = 1e-4) -> float:
        return diferencia_central(f_expr, x, h)

    @staticmethod
    def aitken_accelerator(secuencia: Sequence[float]) -> float:
        return aitken_delta_cuadrado(secuencia)

    @staticmethod
    def aitken_punto_fijo(
        g_expr: str, x0: float, tolerancia: float = 1e-6, max_iter: int = 100
    ) -> AitkenResult:
        return aitken_desde_punto_fijo(g_expr, x0, tolerancia, max_iter)

    @staticmethod
    def runge_kutta_4(
        ode_expr: str,
        t0: float,
        y0: float,
        h: float,
        pasos: int,
    ) -> List[RK4Step]:
        return runge_kutta_4(ode_expr, t0, y0, h, pasos)
