"""Aplicacion de consola para simulacion de modelos matematicos."""

from __future__ import annotations

from typing import List, Tuple

from modelos import crecimiento_logistico, interpolacion_lagrange, metodo_punto_fijo


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
    g_expr = input("Ingresa g(x): ").strip()
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
    print("3) Modelo Logistico Discreto")
    print("0) Salir")


def main() -> None:
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opcion: ").strip()

        if opcion == "1":
            ejecutar_lagrange()
        elif opcion == "2":
            ejecutar_punto_fijo()
        elif opcion == "3":
            ejecutar_logistico()
        elif opcion == "0":
            print("Hasta luego.")
            break
        else:
            print("Opcion invalida.")


if __name__ == "__main__":
    main()
