"""Aplicacion de consola para simulacion de modelos matematicos."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from modelos import (
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
        elif opcion == "0":
            print("Hasta luego.")
            break
        else:
            print("Opcion invalida.")


if __name__ == "__main__":
    main()
