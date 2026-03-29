"""Aplicacion de consola para simulacion de modelos matematicos."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from modelos import (
    aitken_desde_punto_fijo,
    aitken_delta_cuadrado,
    biseccion,
    crecimiento_logistico,
    diferencia_central,
    interpolacion_lagrange,
    metodo_punto_fijo,
    newton_raphson,
    runge_kutta_4,
)


def leer_float(mensaje: str) -> float:
    while True:
        valor = input(mensaje).strip()
        try:
            return float(valor)
        except ValueError:
            print("Entrada invalida. Debes escribir un numero.")


def leer_float_opcional(mensaje: str, valor_por_defecto: float) -> float:
    while True:
        valor = input(mensaje).strip()
        if not valor:
            return valor_por_defecto
        try:
            return float(valor)
        except ValueError:
            print("Entrada invalida. Debes escribir un numero.")


def leer_int(mensaje: str, minimo: int | None = None) -> int:
    while True:
        valor = input(mensaje).strip()
        try:
            numero = int(valor)
            if minimo is not None and numero < minimo:
                print(f"El valor debe ser mayor o igual a {minimo}.")
                continue
            return numero
        except ValueError:
            print("Entrada invalida. Debes escribir un entero.")


def leer_int_opcional(mensaje: str, valor_por_defecto: int, minimo: int = 1) -> int:
    while True:
        valor = input(mensaje).strip()
        if not valor:
            return valor_por_defecto
        try:
            numero = int(valor)
            if numero < minimo:
                print(f"El valor debe ser mayor o igual a {minimo}.")
                continue
            return numero
        except ValueError:
            print("Entrada invalida. Debes escribir un entero.")


def leer_expresion(mensaje: str) -> str:
    while True:
        expr = input(mensaje).strip()
        if expr:
            return expr
        print("La expresion no puede estar vacia.")


def solicitar_puntos() -> List[Tuple[float, float]]:
    cantidad = leer_int("Cantidad de puntos (>=2): ", minimo=2)
    puntos: List[Tuple[float, float]] = []
    for i in range(cantidad):
        x = leer_float(f"Punto {i + 1} - x: ")
        y = leer_float(f"Punto {i + 1} - y: ")
        puntos.append((x, y))
    return puntos


def ejecutar_lagrange() -> None:
    print("\n--- Interpolacion de Lagrange ---")
    puntos = solicitar_puntos()
    x_eval = leer_float("Valor x a interpolar: ")
    try:
        y_interp = interpolacion_lagrange(puntos, x_eval)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"Resultado interpolado: P({x_eval}) = {y_interp}")


def ejecutar_punto_fijo() -> None:
    print("\n--- Metodo de Punto Fijo ---")
    print("Ejemplo de g(x): cos(x), exp(-x), (x + 2/x) / 2")
    g_expr = leer_expresion("Ingresa g(x): ")
    x0 = leer_float("Valor inicial x0: ")
    tolerancia = leer_float("Tolerancia (ej. 1e-6): ")
    max_iter = leer_int("Maximo de iteraciones: ", minimo=1)

    try:
        resultado = metodo_punto_fijo(g_expr, x0, tolerancia, max_iter)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print("\nIteraciones:")
    print("it\t x_anterior\t x_actual\t error")
    for paso in resultado.pasos:
        print(
            f"{paso.iteracion}\t {paso.x_anterior:.8f}\t {paso.x_actual:.8f}\t {paso.error:.3e}"
        )

    if resultado.convergio:
        print(f"\nConvergio a x = {resultado.aproximacion:.10f}")
    else:
        print(
            "\nNo convergio en el maximo de iteraciones. "
            f"Ultima aproximacion: {resultado.aproximacion:.10f}"
        )


def ejecutar_biseccion() -> None:
    print("\n--- Metodo de Biseccion ---")
    print("Ejemplo de f(x): x**3 - x - 4")
    f_expr = leer_expresion("Ingresa f(x): ")
    a = leer_float("Extremo izquierdo a: ")
    b = leer_float("Extremo derecho b: ")
    tolerancia = leer_float("Tolerancia (ej. 1e-6): ")
    max_iter = leer_int("Maximo de iteraciones: ", minimo=1)

    try:
        resultado = biseccion(f_expr, a, b, tolerancia, max_iter)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print("\nIteraciones:")
    print("it\t a\t b\t c\t f(c)\t error_intervalo")
    for paso in resultado.pasos:
        print(
            f"{paso.iteracion}\t {paso.a:.8f}\t {paso.b:.8f}\t {paso.c:.8f}\t "
            f"{paso.fc:.3e}\t {paso.error_intervalo:.3e}"
        )

    if resultado.convergio:
        print(f"\nConvergio a x = {resultado.aproximacion:.10f}")
    else:
        print(
            "\nNo convergio en el maximo de iteraciones. "
            f"Ultima aproximacion: {resultado.aproximacion:.10f}"
        )


def ejecutar_newton() -> None:
    print("\n--- Metodo de Newton-Raphson ---")
    print("Ejemplo: f(x)=x**3 - x - 4, f'(x)=3*x**2 - 1")
    f_expr = leer_expresion("Ingresa f(x): ")
    df_expr = leer_expresion("Ingresa f'(x): ")
    x0 = leer_float("Valor inicial x0: ")
    tolerancia = leer_float("Tolerancia (ej. 1e-6): ")
    max_iter = leer_int("Maximo de iteraciones: ", minimo=1)

    try:
        resultado = newton_raphson(f_expr, df_expr, x0, tolerancia, max_iter)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print("\nIteraciones:")
    print("it\t x_anterior\t x_actual\t f(x_actual)\t error")
    for paso in resultado.pasos:
        print(
            f"{paso.iteracion}\t {paso.x_anterior:.8f}\t {paso.x_actual:.8f}\t "
            f"{paso.fx_actual:.3e}\t {paso.error:.3e}"
        )

    if resultado.convergio:
        print(f"\nConvergio a x = {resultado.aproximacion:.10f}")
    else:
        print(
            "\nNo convergio en el maximo de iteraciones. "
            f"Ultima aproximacion: {resultado.aproximacion:.10f}"
        )


def ejecutar_diferencia_central() -> None:
    print("\n--- Derivacion Numerica (Diferencia Central) ---")
    print("Ejemplo de f(x): sin(x), exp(x), x**2 + 2*x")
    f_expr = leer_expresion("Ingresa f(x): ")
    x = leer_float("Valor donde derivar x: ")
    h = leer_float("Paso h (ej. 1e-4): ")
    try:
        derivada = diferencia_central(f_expr, x, h)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"f'({x}) = {derivada}")


def _solicitar_secuencia() -> Sequence[float]:
    cantidad = leer_int("Cantidad de terminos (>=3): ", minimo=3)
    secuencia: List[float] = []
    for i in range(cantidad):
        secuencia.append(leer_float(f"x_{i}: "))
    return secuencia


def ejecutar_aitken() -> None:
    print("\n--- Aceleracion Delta-Cuadrado de Aitken ---")
    print("Ingresa una secuencia x_n; se usaran los ultimos 3 terminos.")
    secuencia = _solicitar_secuencia()
    try:
        acelerado = aitken_delta_cuadrado(secuencia)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"Valor acelerado de Aitken: {acelerado}")


def ejecutar_rk4() -> None:
    print("\n--- Runge-Kutta de Orden 4 (RK4) ---")
    print("La EDO debe escribirse como y' = f(t, y). Ejemplo: 0.1*y")
    ode_expr = leer_expresion("Ingresa f(t, y): ")
    t0 = leer_float("Tiempo inicial t0: ")
    y0 = leer_float("Condicion inicial y0: ")
    h = leer_float("Paso h: ")
    pasos = leer_int("Cantidad de pasos: ", minimo=1)

    try:
        trayectoria = runge_kutta_4(ode_expr, t0, y0, h, pasos)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print("\nTrayectoria:")
    print("paso\t t\t y")
    for punto in trayectoria:
        print(f"{punto.paso}\t {punto.t:.8f}\t {punto.y:.8f}")


def ejecutar_logistico() -> None:
    print("\n--- Modelo Logistico Discreto ---")
    print("Modelo: x_(n+1) = r * x_n * (1 - x_n / K)")
    r = leer_float("Parametro r: ")
    k = leer_float("Capacidad de carga K: ")
    x0 = leer_float("Poblacion inicial x0: ")
    pasos = leer_int("Cantidad de pasos: ", minimo=1)

    try:
        serie = crecimiento_logistico(r, k, x0, pasos)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print("\nEvolucion:")
    for idx, valor in enumerate(serie):
        print(f"x_{idx} = {valor}")


_BISECCION_PRESETS = {
    1: ("sqrt(x) - cos(x) = 0 en [0,1]", "sqrt(x) - cos(x)", 0.0, 1.0, 1e-3, 100),
    2: ("x - 2**(-x) = 0 en [0,1]", "x - 2**(-x)", 0.0, 1.0, 1e-3, 100),
    3: ("exp(x) - x**2 + 3*x - 2 = 0 en [0,1]", "exp(x) - x**2 + 3*x - 2", 0.0, 1.0, 1e-3, 100),
    4: ("2*x*cos(x) - (x+1)**2 = 0 en [-3,-2]", "2*x*cos(x) - (x+1)**2", -3.0, -2.0, 1e-3, 100),
    5: ("2*x*cos(x) - (x+1)**2 = 0 en [-1,0]", "2*x*cos(x) - (x+1)**2", -1.0, 0.0, 1e-3, 100),
    6: ("x*cos(x) - 2*x**2 + 3*x - 1 = 0 en [0.2,0.3]", "x*cos(x) - 2*x**2 + 3*x - 1", 0.2, 0.3, 1e-3, 100),
    7: ("x*cos(x) - 2*x**2 + 3*x - 1 = 0 en [1.2,1.3]", "x*cos(x) - 2*x**2 + 3*x - 1", 1.2, 1.3, 1e-3, 100),
    8: ("x**4 - 2*x**3 - 4*x**2 + 4*x + 4 en [-2,-1]", "x**4 - 2*x**3 - 4*x**2 + 4*x + 4", -2.0, -1.0, 1e-2, 100),
    9: ("x**4 - 2*x**3 - 4*x**2 + 4*x + 4 en [0,2]", "x**4 - 2*x**3 - 4*x**2 + 4*x + 4", 0.0, 2.0, 1e-2, 100),
    10: ("x**4 - 2*x**3 - 4*x**2 + 4*x + 4 en [2,3]", "x**4 - 2*x**3 - 4*x**2 + 4*x + 4", 2.0, 3.0, 1e-2, 100),
}

_PUNTO_FIJO_PRESETS = {
    1: ("2*exp(x**2)-5*x=0, usar g(x)=0.4*exp(x**2), x0=0", "0.4*exp(x**2)", 0.0, 1e-6, 100),
    2: ("cos(x)-x=0, usar g(x)=cos(x), x0=1", "cos(x)", 1.0, 1e-6, 100),
    3: ("exp(-x)-x=0, usar g(x)=exp(-x), x0=0", "exp(-x)", 0.0, 1e-6, 100),
    4: ("x**3-x-1=0, usar g(x)=(x+1)**(1/3), x0=1", "(x+1)**(1/3)", 1.0, 1e-6, 100),
    5: ("pi+0.5*sin(x**2)-x=0, usar g(x)=pi+0.5*sin(x**2), x0=0", "pi + 0.5*sin(x**2)", 0.0, 1e-6, 100),
    6: ("Aproximar sqrt(3), g(x)=(x+3/x)/2, x0=1", "(x + 3/x)/2", 1.0, 1e-4, 100),
    7: ("g(x)=2**(-x), x0=0.5", "2**(-x)", 0.5, 1e-6, 100),
}

_NEWTON_PRESETS = {
    1: ("f(x)=(x-1)**2, x0=0", "(x-1)**2", "2*(x-1)", 0.0, 1e-6, 100),
    2: ("f(x)=x**3-2*x-5, x0=1.5", "x**3 - 2*x - 5", "3*x**2 - 2", 1.5, 1e-6, 100),
    3: ("f(x)=x**5-x-1, x0=1", "x**5 - x - 1", "5*x**4 - 1", 1.0, 1e-6, 100),
    4: ("Aproximar 2**(1/6): f(x)=x**6-2, x0=1", "x**6 - 2", "6*x**5", 1.0, 1e-8, 100),
    5: ("f(x)=exp(x)+x**2-4, x0=0.5", "exp(x) + x**2 - 4", "exp(x) + 2*x", 0.5, 1e-6, 100),
    6: ("f(x)=x**2-3*x-4, x0=8", "x**2 - 3*x - 4", "2*x - 3", 8.0, 1e-6, 100),
    7: ("f(x)=log(x)-1, x0=2", "log(x) - 1", "1/x", 2.0, 1e-6, 100),
    8: ("f(x)=x**4-16, x0=2", "x**4 - 16", "4*x**3", 2.0, 1e-6, 100),
    9: ("f(x)=x**3-2*x+1, x0=-1.5", "x**3 - 2*x + 1", "3*x**2 - 2", -1.5, 1e-6, 100),
    10: ("f(x)=exp(3*x)-4, x0=0", "exp(3*x) - 4", "3*exp(3*x)", 0.0, 1e-6, 100),
}

_AITKEN_PRESETS = {
    1: ("g(x)=cos(x), x0=0.5", "cos(x)", 0.5, 1e-6, 50),
    2: ("g(x)=exp(-x), x0=1", "exp(-x)", 1.0, 1e-6, 50),
    3: ("g(x)=sqrt(3*x-2), x0=2", "sqrt(3*x - 2)", 2.0, 1e-6, 50),
    4: ("g(x)=log(x+1), x0=0.5", "log(x + 1)", 0.5, 1e-6, 50),
    5: ("g(x)=1-x**3, x0=0.5", "1 - x**3", 0.5, 1e-6, 50),
    6: ("g(x)=0.5*(x**2-3), x0=0.5", "0.5*(x**2 - 3)", 0.5, 1e-6, 50),
}

_LAGRANGE_PRESETS = {
    1: ("Puntos (1,1), (2,4), (3,9), evaluar en x=1.5", [(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)], 1.5),
    2: ("Puntos (0,1), (1,3), (2,2), (3,5), evaluar en x=1.5", [(0.0, 1.0), (1.0, 3.0), (2.0, 2.0), (3.0, 5.0)], 1.5),
    3: ("x=[0,1,2,3,4], y=[1,2,0,2,3], evaluar en x=2.5", [(0.0, 1.0), (1.0, 2.0), (2.0, 0.0), (3.0, 2.0), (4.0, 3.0)], 2.5),
    4: ("x=[0,1,2], y=[1,3,0], evaluar en x=1.5", [(0.0, 1.0), (1.0, 3.0), (2.0, 0.0)], 1.5),
    5: ("Aproximar f(x)=1/x con nodos 2,2.5,4.5 en x=3", [(2.0, 0.5), (2.5, 0.4), (4.5, 1 / 4.5)], 3.0),
    6: ("f(x)=2*sin(pi*x/6), nodos 1,2,3; aproximar f(4)", [(1.0, 2.0 * math.sin(math.pi / 6)), (2.0, 2.0 * math.sin(2.0 * math.pi / 6)), (3.0, 2.0 * math.sin(3.0 * math.pi / 6))], 4.0),
}

_DIFERENCIA_PRESETS = {
    1: ("f(x)=sin(x), x=[0,0.1,...,0.5], h=0.1", "sin(x)", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 0.1),
    2: ("f(x)=exp(x), x=[0,0.1,...,0.5], h=0.1", "exp(x)", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 0.1),
    3: ("f(x)=x**3-x, derivada en x=1 con h=0.1", "x**3 - x", [1.0], 0.1),
    4: ("f(x)=exp(x)*sin(x), derivada en x=1 con h=0.01", "exp(x)*sin(x)", [1.0], 0.01),
}

_RK4_PRESETS = {
    1: ("y'=y+t**2, y(0)=1, h=0.1, 0<=t<=1", "y + t**2", 0.0, 1.0, 0.1, 10),
    2: ("y'=y*sin(t), y(0)=2, h=pi/10, 0<=t<=pi", "y*sin(t)", 0.0, 2.0, math.pi / 10.0, 10),
    3: ("y'=2*t+3*y, y(0)=0, h=0.2, 0<=t<=1", "2*t + 3*y", 0.0, 0.0, 0.2, 5),
    4: ("y'=t-y**2, y(0)=1, h=0.2, 0<=t<=2", "t - y**2", 0.0, 1.0, 0.2, 10),
    5: ("y'=exp(-t)-y, y(0)=0, h=0.1, 0<=t<=1", "exp(-t) - y", 0.0, 0.0, 0.1, 10),
    6: ("y'=1/(1+t**2)-y, y(0)=1, h=0.5, 0<=t<=2", "1/(1+t**2) - y", 0.0, 1.0, 0.5, 4),
}


def _elegir_preset(
    titulo: str, presets: dict[int, tuple]
) -> tuple | None:
    print(f"\n--- {titulo} ---")
    for key, preset in presets.items():
        print(f"{key}) {preset[0]}")
    print("0) Volver")
    opcion = leer_int("Elegi un ejercicio: ", minimo=0)
    if opcion == 0:
        return None
    if opcion not in presets:
        print("Ejercicio invalido.")
        return None
    return presets[opcion]


def _ejercicio_biseccion() -> None:
    seleccionado = _elegir_preset("Ejercicios PDF - Biseccion", _BISECCION_PRESETS)
    if seleccionado is None:
        return
    _, f_expr, a, b, tol_default, iter_default = seleccionado
    tolerancia = leer_float_opcional(
        f"Tolerancia (Enter={tol_default}): ", tol_default
    )
    max_iter = leer_int_opcional(
        f"Max iteraciones (Enter={iter_default}): ", iter_default
    )
    try:
        resultado = biseccion(f_expr, a, b, tolerancia, max_iter)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"\nAprox raiz: {resultado.aproximacion:.10f} | convergio={resultado.convergio}")


def _ejercicio_punto_fijo() -> None:
    seleccionado = _elegir_preset(
        "Ejercicios PDF - Punto Fijo", _PUNTO_FIJO_PRESETS
    )
    if seleccionado is None:
        return
    _, g_expr, x0, tol_default, iter_default = seleccionado
    tolerancia = leer_float_opcional(
        f"Tolerancia (Enter={tol_default}): ", tol_default
    )
    max_iter = leer_int_opcional(
        f"Max iteraciones (Enter={iter_default}): ", iter_default
    )
    try:
        resultado = metodo_punto_fijo(g_expr, x0, tolerancia, max_iter)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"\nAprox raiz: {resultado.aproximacion:.10f} | convergio={resultado.convergio}")


def _ejercicio_newton() -> None:
    seleccionado = _elegir_preset("Ejercicios PDF - Newton", _NEWTON_PRESETS)
    if seleccionado is None:
        return
    _, f_expr, df_expr, x0, tol_default, iter_default = seleccionado
    tolerancia = leer_float_opcional(
        f"Tolerancia (Enter={tol_default}): ", tol_default
    )
    max_iter = leer_int_opcional(
        f"Max iteraciones (Enter={iter_default}): ", iter_default
    )
    try:
        resultado = newton_raphson(f_expr, df_expr, x0, tolerancia, max_iter)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"\nAprox raiz: {resultado.aproximacion:.10f} | convergio={resultado.convergio}")


def _ejercicio_aitken() -> None:
    print("\n--- Ejercicios PDF - Aitken ---")
    print("1) Desde secuencia manual")
    print("2) Desde iteracion de punto fijo (presets del PDF)")
    print("0) Volver")
    opcion = leer_int("Elegi una opcion: ", minimo=0)
    if opcion == 0:
        return
    if opcion == 1:
        secuencia = _solicitar_secuencia()
        try:
            acelerado = aitken_delta_cuadrado(secuencia)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        print(f"Valor acelerado: {acelerado}")
        return
    if opcion != 2:
        print("Opcion invalida.")
        return

    seleccionado = _elegir_preset("Aitken con g(x) y x0", _AITKEN_PRESETS)
    if seleccionado is None:
        return
    _, g_expr, x0, tol_default, iter_default = seleccionado
    tolerancia = leer_float_opcional(
        f"Tolerancia (Enter={tol_default}): ", tol_default
    )
    max_iter = leer_int_opcional(
        f"Max iteraciones (Enter={iter_default}): ", iter_default
    )
    try:
        resultado = aitken_desde_punto_fijo(g_expr, x0, tolerancia, max_iter)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(
        f"\nAprox Aitken: {resultado.aproximacion:.10f} | convergio={resultado.convergio}"
    )


def _ejercicio_lagrange() -> None:
    seleccionado = _elegir_preset("Ejercicios PDF - Lagrange", _LAGRANGE_PRESETS)
    if seleccionado is None:
        return
    _, puntos, x_eval = seleccionado
    try:
        valor = interpolacion_lagrange(puntos, x_eval)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"\nP({x_eval}) = {valor}")


def _ejercicio_diferencias() -> None:
    seleccionado = _elegir_preset(
        "Ejercicios PDF - Diferencias Finitas", _DIFERENCIA_PRESETS
    )
    if seleccionado is None:
        return
    _, f_expr, xs, h = seleccionado
    print("\nResultados f'(x) aproximada:")
    for x in xs:
        try:
            derivada = diferencia_central(f_expr, x, h)
        except ValueError as exc:
            print(f"x={x}: error {exc}")
            continue
        print(f"x={x:.4f} -> {derivada:.10f}")


def _ejercicio_rk4() -> None:
    seleccionado = _elegir_preset("Ejercicios PDF - RK4", _RK4_PRESETS)
    if seleccionado is None:
        return
    _, ode_expr, t0, y0, h, pasos = seleccionado
    try:
        trayectoria = runge_kutta_4(ode_expr, t0, y0, h, pasos)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print("\nTrayectoria:")
    print("paso\t t\t y")
    for punto in trayectoria:
        print(f"{punto.paso}\t {punto.t:.8f}\t {punto.y:.8f}")


def ejecutar_ejercicios_pdf() -> None:
    while True:
        print("\n==============================")
        print(" Ejercicios guiados del PDF")
        print("==============================")
        print("1) Biseccion")
        print("2) Punto Fijo")
        print("3) Newton-Raphson")
        print("4) Aitken")
        print("5) Lagrange")
        print("6) Diferencias Finitas")
        print("7) RK4")
        print("0) Volver")
        opcion = input("Selecciona una opcion: ").strip()

        if opcion == "1":
            _ejercicio_biseccion()
        elif opcion == "2":
            _ejercicio_punto_fijo()
        elif opcion == "3":
            _ejercicio_newton()
        elif opcion == "4":
            _ejercicio_aitken()
        elif opcion == "5":
            _ejercicio_lagrange()
        elif opcion == "6":
            _ejercicio_diferencias()
        elif opcion == "7":
            _ejercicio_rk4()
        elif opcion == "0":
            return
        else:
            print("Opcion invalida.")


def mostrar_menu() -> None:
    print("\n==============================")
    print(" Simulador de Modelos Matematicos")
    print("==============================")
    print("1) Interpolacion de Lagrange")
    print("2) Metodo de Punto Fijo")
    print("3) Metodo de Biseccion")
    print("4) Metodo de Newton-Raphson")
    print("5) Diferencia Central (Derivacion numerica)")
    print("6) Aceleracion de Aitken")
    print("7) Runge-Kutta 4 (EDO)")
    print("8) Modelo Logistico Discreto")
    print("9) Resolver ejercicios del PDF (modo guiado)")
    print("0) Salir")


def main() -> None:
    while True:
        mostrar_menu()
        try:
            opcion = input("Selecciona una opcion: ").strip()
        except EOFError:
            print("\nEntrada finalizada. Saliendo del simulador.")
            break

        if opcion == "1":
            ejecutar_lagrange()
        elif opcion == "2":
            ejecutar_punto_fijo()
        elif opcion == "3":
            ejecutar_biseccion()
        elif opcion == "4":
            ejecutar_newton()
        elif opcion == "5":
            ejecutar_diferencia_central()
        elif opcion == "6":
            ejecutar_aitken()
        elif opcion == "7":
            ejecutar_rk4()
        elif opcion == "8":
            ejecutar_logistico()
        elif opcion == "9":
            ejecutar_ejercicios_pdf()
        elif opcion == "0":
            print("Hasta luego.")
            break
        else:
            print("Opcion invalida.")


if __name__ == "__main__":
    main()
