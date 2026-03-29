"""Interfaz web (Streamlit) para el simulador de metodos numericos."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import streamlit as st

from modelos import (
    aitken_desde_punto_fijo,
    aitken_delta_cuadrado,
    biseccion,
    diferencia_central,
    interpolacion_lagrange,
    metodo_punto_fijo,
    newton_raphson,
    runge_kutta_4,
)


def _parsear_puntos(raw: str) -> List[Tuple[float, float]]:
    """Convierte texto tipo '1,2; 2,4; 3,9' en lista de puntos."""
    puntos: List[Tuple[float, float]] = []
    for bloque in raw.split(";"):
        bloque = bloque.strip()
        if not bloque:
            continue
        partes = [x.strip() for x in bloque.split(",")]
        if len(partes) != 2:
            raise ValueError(
                "Formato invalido. Usa pares tipo x,y separados por ';'."
            )
        puntos.append((float(partes[0]), float(partes[1])))
    if len(puntos) < 2:
        raise ValueError("Debes ingresar al menos dos puntos.")
    return puntos


def _parsear_secuencia(raw: str) -> List[float]:
    """Convierte texto tipo '1, 0.75, 0.6875, 0.671875' a lista."""
    valores: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        valores.append(float(token))
    if len(valores) < 3:
        raise ValueError("Debes ingresar al menos tres terminos.")
    return valores


def _mostrar_titulo() -> None:
    st.set_page_config(page_title="Modelado y Simulacion", page_icon="📈", layout="wide")
    st.title("📈 Simulador de Modelos Matematicos")
    st.caption("Interfaz web interactiva para resolver ejercicios numericos.")


def _panel_biseccion() -> None:
    st.subheader("Metodo de Biseccion")
    col1, col2 = st.columns(2)
    with col1:
        f_expr = st.text_input("f(x)", value="sqrt(x) - cos(x)")
        a = st.number_input("a", value=0.0)
        b = st.number_input("b", value=1.0)
    with col2:
        tolerancia = st.number_input("Tolerancia", value=1e-6, format="%.10f")
        max_iter = st.number_input("Max iteraciones", min_value=1, value=100, step=1)

    if st.button("Resolver Biseccion", use_container_width=True):
        try:
            resultado = biseccion(f_expr, a, b, float(tolerancia), int(max_iter))
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success(
            f"Aprox raiz: {resultado.aproximacion:.10f} | convergio={resultado.convergio}"
        )
        if resultado.pasos:
            st.dataframe([asdict(p) for p in resultado.pasos], use_container_width=True)


def _panel_punto_fijo() -> None:
    st.subheader("Metodo de Punto Fijo")
    col1, col2 = st.columns(2)
    with col1:
        g_expr = st.text_input("g(x)", value="cos(x)")
        x0 = st.number_input("x0", value=0.5)
    with col2:
        tolerancia = st.number_input(
            "Tolerancia PF", value=1e-6, format="%.10f", key="tol_pf"
        )
        max_iter = st.number_input(
            "Max iter PF", min_value=1, value=100, step=1, key="iter_pf"
        )

    if st.button("Resolver Punto Fijo", use_container_width=True):
        try:
            resultado = metodo_punto_fijo(g_expr, x0, float(tolerancia), int(max_iter))
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(
            f"Aprox raiz: {resultado.aproximacion:.10f} | convergio={resultado.convergio}"
        )
        st.dataframe([asdict(p) for p in resultado.pasos], use_container_width=True)


def _panel_newton() -> None:
    st.subheader("Metodo de Newton-Raphson")
    col1, col2 = st.columns(2)
    with col1:
        f_expr = st.text_input("f(x) Newton", value="x**3 - 2*x - 5")
        df_expr = st.text_input("f'(x)", value="3*x**2 - 2")
        x0 = st.number_input("x0 Newton", value=1.5)
    with col2:
        tolerancia = st.number_input(
            "Tolerancia Newton", value=1e-6, format="%.10f", key="tol_newton"
        )
        max_iter = st.number_input(
            "Max iter Newton", min_value=1, value=100, step=1, key="iter_newton"
        )

    if st.button("Resolver Newton", use_container_width=True):
        try:
            resultado = newton_raphson(
                f_expr, df_expr, x0, float(tolerancia), int(max_iter)
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(
            f"Aprox raiz: {resultado.aproximacion:.10f} | convergio={resultado.convergio}"
        )
        st.dataframe([asdict(p) for p in resultado.pasos], use_container_width=True)


def _panel_lagrange() -> None:
    st.subheader("Interpolacion de Lagrange")
    raw_puntos = st.text_area(
        "Puntos (formato x,y; x,y; ...)",
        value="1,1; 2,4; 3,9",
    )
    x_eval = st.number_input("x a interpolar", value=1.5)

    if st.button("Interpolar", use_container_width=True):
        try:
            puntos = _parsear_puntos(raw_puntos)
            valor = interpolacion_lagrange(puntos, x_eval)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(f"P({x_eval}) = {valor}")


def _panel_diferencia_central() -> None:
    st.subheader("Derivacion Numerica (Diferencia Central)")
    col1, col2 = st.columns(2)
    with col1:
        f_expr = st.text_input("f(x) dif", value="sin(x)")
        x = st.number_input("x derivar", value=0.3)
    with col2:
        h = st.number_input("h", value=0.01, format="%.10f")

    if st.button("Derivar", use_container_width=True):
        try:
            derivada = diferencia_central(f_expr, x, float(h))
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(f"f'({x}) ≈ {derivada}")


def _panel_aitken() -> None:
    st.subheader("Aceleracion de Aitken")
    modo = st.radio("Modo", ["Secuencia manual", "Desde punto fijo"], horizontal=True)

    if modo == "Secuencia manual":
        raw = st.text_input("Secuencia", value="1, 0.75, 0.6875, 0.671875")
        if st.button("Aplicar Aitken (manual)", use_container_width=True):
            try:
                secuencia = _parsear_secuencia(raw)
                valor = aitken_delta_cuadrado(secuencia)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.success(f"Valor acelerado: {valor}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            g_expr = st.text_input("g(x) Aitken", value="exp(-x)")
            x0 = st.number_input("x0 Aitken", value=1.0)
        with col2:
            tolerancia = st.number_input(
                "Tolerancia Aitken", value=1e-6, format="%.10f", key="tol_aitken"
            )
            max_iter = st.number_input(
                "Max iter Aitken",
                min_value=1,
                value=50,
                step=1,
                key="iter_aitken",
            )
        if st.button("Aplicar Aitken (punto fijo)", use_container_width=True):
            try:
                resultado = aitken_desde_punto_fijo(
                    g_expr, x0, float(tolerancia), int(max_iter)
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            st.success(
                f"Aprox Aitken: {resultado.aproximacion:.10f} "
                f"| convergio={resultado.convergio}"
            )
            st.dataframe([asdict(p) for p in resultado.pasos], use_container_width=True)


def _panel_rk4() -> None:
    st.subheader("Runge-Kutta de Orden 4 (RK4)")
    col1, col2 = st.columns(2)
    with col1:
        ode_expr = st.text_input("f(t, y)", value="y + t**2")
        t0 = st.number_input("t0", value=0.0)
        y0 = st.number_input("y0", value=1.0)
    with col2:
        h = st.number_input("h RK4", value=0.1, format="%.10f")
        pasos = st.number_input("Pasos", min_value=1, value=10, step=1)

    if st.button("Simular RK4", use_container_width=True):
        try:
            trayectoria = runge_kutta_4(ode_expr, t0, y0, float(h), int(pasos))
        except ValueError as exc:
            st.error(str(exc))
            return
        tabla = [asdict(p) for p in trayectoria]
        st.dataframe(tabla, use_container_width=True)
        st.line_chart({"y": [p["y"] for p in tabla]}, use_container_width=True)


def main() -> None:
    _mostrar_titulo()
    opcion = st.sidebar.selectbox(
        "Elegi un metodo",
        [
            "Biseccion",
            "Punto Fijo",
            "Newton-Raphson",
            "Lagrange",
            "Diferencia Central",
            "Aitken",
            "RK4",
        ],
    )

    if opcion == "Biseccion":
        _panel_biseccion()
    elif opcion == "Punto Fijo":
        _panel_punto_fijo()
    elif opcion == "Newton-Raphson":
        _panel_newton()
    elif opcion == "Lagrange":
        _panel_lagrange()
    elif opcion == "Diferencia Central":
        _panel_diferencia_central()
    elif opcion == "Aitken":
        _panel_aitken()
    else:
        _panel_rk4()


if __name__ == "__main__":
    main()
