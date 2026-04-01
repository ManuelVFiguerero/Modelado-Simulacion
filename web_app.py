"""Interfaz web (Streamlit) para el simulador de metodos numericos."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import plotly.graph_objects as go
import streamlit as st

from modelos import (
    aitken_desde_punto_fijo,
    aitken_delta_cuadrado,
    biseccion,
    diferencia_central,
    gauss_legendre_cuadratura,
    interpolacion_lagrange,
    metodo_punto_fijo,
    newton_raphson,
    rectangulo_medio_compuesto,
    runge_kutta_4,
    simpson_13_compuesto,
    simpson_38_compuesto,
    trapecio_compuesto,
)


def _eval_expr_points(expr: str, xs: List[float]) -> List[float]:
    """Evalua expresiones con la misma whitelist usada por modelos.py."""
    ys: List[float] = []
    safe_math = {
        "sin": __import__("math").sin,
        "cos": __import__("math").cos,
        "tan": __import__("math").tan,
        "asin": __import__("math").asin,
        "acos": __import__("math").acos,
        "atan": __import__("math").atan,
        "exp": __import__("math").exp,
        "log": __import__("math").log,
        "log10": __import__("math").log10,
        "sqrt": __import__("math").sqrt,
        "fabs": __import__("math").fabs,
        "sinh": __import__("math").sinh,
        "cosh": __import__("math").cosh,
        "tanh": __import__("math").tanh,
        "floor": __import__("math").floor,
        "ceil": __import__("math").ceil,
        "pi": __import__("math").pi,
        "e": __import__("math").e,
        "abs": abs,
        "pow": pow,
    }
    for x in xs:
        ys.append(float(eval(expr, {"__builtins__": {}}, {**safe_math, "x": x})))
    return ys


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
            tabla = [asdict(p) for p in resultado.pasos]
            st.dataframe(tabla, use_container_width=True)

            fig_conv = go.Figure()
            fig_conv.add_trace(
                go.Scatter(
                    x=[fila["iteracion"] for fila in tabla],
                    y=[fila["error_intervalo"] for fila in tabla],
                    mode="lines+markers",
                    name="error_intervalo",
                )
            )
            fig_conv.update_layout(
                title="Convergencia de Biseccion",
                xaxis_title="Iteracion",
                yaxis_title="Error de intervalo",
            )
            st.plotly_chart(fig_conv, use_container_width=True)

            try:
                xs = [a + (b - a) * i / 200 for i in range(201)]
                ys = _eval_expr_points(f_expr, xs)
                fig_fx = go.Figure()
                fig_fx.add_trace(
                    go.Scatter(x=xs, y=ys, mode="lines", name="f(x)")
                )
                fig_fx.add_hline(y=0, line_dash="dash")
                fig_fx.add_vline(x=resultado.aproximacion, line_dash="dot")
                fig_fx.update_layout(
                    title="Funcion y raiz aproximada",
                    xaxis_title="x",
                    yaxis_title="f(x)",
                )
                st.plotly_chart(fig_fx, use_container_width=True)
            except Exception:
                st.info("No se pudo graficar f(x) en el intervalo elegido.")


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
        tabla = [asdict(p) for p in resultado.pasos]
        st.dataframe(tabla, use_container_width=True)

        fig_err = go.Figure()
        fig_err.add_trace(
            go.Scatter(
                x=[fila["iteracion"] for fila in tabla],
                y=[fila["error"] for fila in tabla],
                mode="lines+markers",
                name="error",
            )
        )
        fig_err.update_layout(
            title="Convergencia de Punto Fijo",
            xaxis_title="Iteracion",
            yaxis_title="|x_n+1 - x_n|",
        )
        st.plotly_chart(fig_err, use_container_width=True)


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
        tabla = [asdict(p) for p in resultado.pasos]
        st.dataframe(tabla, use_container_width=True)

        fig_err = go.Figure()
        fig_err.add_trace(
            go.Scatter(
                x=[fila["iteracion"] for fila in tabla],
                y=[fila["error"] for fila in tabla],
                mode="lines+markers",
                name="error",
            )
        )
        fig_err.update_layout(
            title="Convergencia de Newton-Raphson",
            xaxis_title="Iteracion",
            yaxis_title="Error",
        )
        st.plotly_chart(fig_err, use_container_width=True)


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

        xs_datos = [p[0] for p in puntos]
        x_min, x_max = min(xs_datos), max(xs_datos)
        margen = (x_max - x_min) * 0.2 if x_max > x_min else 1.0
        xs = [x_min - margen + (x_max - x_min + 2 * margen) * i / 200 for i in range(201)]
        ys = [interpolacion_lagrange(puntos, x) for x in xs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Polinomio"))
        fig.add_trace(
            go.Scatter(
                x=xs_datos,
                y=[p[1] for p in puntos],
                mode="markers",
                name="Nodos",
                marker=dict(size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_eval],
                y=[valor],
                mode="markers",
                name="Punto evaluado",
                marker=dict(size=11, symbol="diamond"),
            )
        )
        fig.update_layout(title="Interpolacion de Lagrange", xaxis_title="x", yaxis_title="y")
        st.plotly_chart(fig, use_container_width=True)


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

        try:
            xs = [x - 5 * float(h) + i * float(h) for i in range(11)]
            ys = _eval_expr_points(f_expr, xs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="f(x)"))
            fig.add_vline(x=x, line_dash="dash")
            fig.update_layout(title="Vecindad de f(x) para derivacion", xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("No se pudo graficar la funcion para esta expresion.")


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
            tabla = [asdict(p) for p in resultado.pasos]
            st.dataframe(tabla, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[fila["iteracion"] for fila in tabla],
                    y=[fila["error"] for fila in tabla],
                    mode="lines+markers",
                    name="error Aitken",
                )
            )
            fig.update_layout(
                title="Convergencia de Aitken",
                xaxis_title="Iteracion",
                yaxis_title="Error",
            )
            st.plotly_chart(fig, use_container_width=True)


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

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[p["t"] for p in tabla],
                y=[p["y"] for p in tabla],
                mode="lines+markers",
                name="y(t)",
            )
        )
        fig.update_layout(title="Trayectoria RK4", xaxis_title="t", yaxis_title="y")
        st.plotly_chart(fig, use_container_width=True)


def _panel_integracion() -> None:
    st.subheader("Integracion Numerica (Newton-Cotes y Gauss)")

    metodo = st.selectbox(
        "Metodo",
        [
            "Trapecio compuesto",
            "Simpson 1/3 compuesto",
            "Simpson 3/8 compuesto",
            "Rectangulo medio compuesto",
            "Cuadratura de Gauss-Legendre",
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        f_expr = st.text_input("f(x) integrar", value="sin(x)")
        a = st.number_input("Limite inferior a", value=0.0)
        b = st.number_input("Limite superior b", value=3.1415926536)
    with col2:
        n_default = 6 if "3/8" in metodo else 4
        n = st.number_input("Subintervalos / puntos n", min_value=1, value=n_default, step=1)

    if st.button("Integrar", use_container_width=True):
        try:
            if metodo == "Trapecio compuesto":
                integral = trapecio_compuesto(f_expr, a, b, int(n))
            elif metodo == "Simpson 1/3 compuesto":
                integral = simpson_13_compuesto(f_expr, a, b, int(n))
            elif metodo == "Simpson 3/8 compuesto":
                integral = simpson_38_compuesto(f_expr, a, b, int(n))
            elif metodo == "Rectangulo medio compuesto":
                integral = rectangulo_medio_compuesto(f_expr, a, b, int(n))
            else:
                integral = gauss_legendre_cuadratura(f_expr, a, b, int(n))
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success(f"Integral aproximada: {integral:.12f}")

        # Visualizacion del integrando en [a,b]
        try:
            xs = [a + (b - a) * i / 300 for i in range(301)]
            ys = _eval_expr_points(f_expr, xs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)"))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(title="Integrando en [a,b]", xaxis_title="x", yaxis_title="f(x)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("No se pudo graficar la funcion en el intervalo.")


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
            "Integracion Numerica",
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
    elif opcion == "Integracion Numerica":
        _panel_integracion()
    else:
        _panel_rk4()


if __name__ == "__main__":
    main()
