"""Interfaz web (Streamlit) para el simulador de metodos numericos."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import List, Tuple

import plotly.graph_objects as go
import streamlit as st

from modelos import (
    aitken_desde_punto_fijo,
    aitken_delta_cuadrado,
    biseccion,
    cuadratura_gauss_legendre,
    diferencia_central,
    evaluar_expresion,
    get_angular_mode,
    integracion_montecarlo,
    interpolacion_lagrange,
    metodo_punto_fijo,
    newton_raphson,
    rectangulo_medio_compuesto,
    runge_kutta_4,
    set_angular_mode,
    simpson_13_compuesto,
    simpson_38_compuesto,
    trapecio_compuesto,
)


_Z_SCORES = {
    0.8: 1.2815515655446004,
    0.9: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
    0.997: 2.9677379253417944,
}


def _eval_expr_points(expr: str, xs: List[float]) -> List[float]:
    """Evalua expresiones usando el motor comun (respeta modo angular)."""
    ys: List[float] = []
    for x in xs:
        ys.append(float(evaluar_expresion(expr, x=x)))
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
    if "angle_mode" not in st.session_state:
        st.session_state["angle_mode"] = "radianes"
    modo = st.sidebar.radio(
        "Modo angular trigonometrico",
        ["Radianes", "Grados"],
        index=0 if st.session_state["angle_mode"] == "radianes" else 1,
    )
    st.session_state["angle_mode"] = "radianes" if modo == "Radianes" else "grados"
    st.sidebar.caption(
        f"Modo activo: {st.session_state['angle_mode']} (sin/cos/tan)"
    )


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
            resultado = newton_raphson(f_expr, df_expr, x0, float(tolerancia), int(max_iter))
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
                resultado = aitken_desde_punto_fijo(g_expr, x0, float(tolerancia), int(max_iter))
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
                integral = cuadratura_gauss_legendre(f_expr, a, b, int(n))
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


def _panel_montecarlo() -> None:
    st.subheader("Integracion por Monte Carlo")
    st.caption(
        "Estimacion de integral definida con muestreo uniforme, error estandar e intervalo de confianza."
    )

    col1, col2 = st.columns(2)
    with col1:
        f_expr = st.text_input("f(x) Monte Carlo", value="exp(-x**2)")
        a = st.number_input("Limite inferior a (MC)", value=0.0)
        b = st.number_input("Limite superior b (MC)", value=1.0)
    with col2:
        n = st.number_input("Muestras n", min_value=100, value=10000, step=100)
        confianza = st.selectbox("Nivel de confianza", [0.8, 0.9, 0.95, 0.99, 0.997], index=2)
        seed_raw = st.text_input("Semilla aleatoria (opcional)", value="")

    if st.button("Integrar con Monte Carlo", use_container_width=True):
        seed = None
        if seed_raw.strip():
            try:
                seed = int(seed_raw.strip())
            except ValueError:
                st.error("La semilla debe ser un entero.")
                return

        try:
            resultado = integracion_montecarlo(
                f_expr,
                a,
                b,
                int(n),
                float(confianza),
                seed,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success(f"Integral estimada: {resultado.estimacion:.12f}")
        st.markdown(
            f"- Desvio muestral: `{resultado.desvio_muestral:.6e}`\n"
            f"- Error estandar: `{resultado.error_estandar:.6e}`\n"
            f"- IC {resultado.confianza*100:.1f}%: "
            f"`[{resultado.ic_bajo:.12f}, {resultado.ic_alto:.12f}]`"
        )

        if resultado.muestras_transformadas:
            valores = resultado.muestras_transformadas
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=valores,
                    nbinsx=40,
                    name="Muestras transformadas",
                    marker_color="#4f81bd",
                )
            )
            fig_hist.add_vline(x=resultado.estimacion, line_dash="dash", line_color="green")
            fig_hist.update_layout(
                title="Distribucion de aportes Monte Carlo",
                xaxis_title="(b-a) * f(U(a,b))",
                yaxis_title="Frecuencia",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def _panel_montecarlo() -> None:
    st.subheader("Integracion por Monte Carlo")
    col1, col2 = st.columns(2)
    with col1:
        f_expr = st.text_input("f(x) Monte Carlo", value="sin(x)")
        a = st.number_input("Limite inferior a (MC)", value=0.0)
        b = st.number_input("Limite superior b (MC)", value=3.1415926536)
        n = st.number_input("Muestras n", min_value=100, value=5000, step=100)
    with col2:
        confianza = st.selectbox(
            "Nivel de confianza",
            [0.8, 0.9, 0.95, 0.99, 0.997],
            index=2,
            format_func=lambda c: f"{c*100:.1f}%",
        )
        seed_str = st.text_input("Semilla aleatoria (opcional)", value="")
        seed = int(seed_str) if seed_str.strip() else None

    if st.button("Integrar con Monte Carlo", use_container_width=True):
        try:
            resultado = integracion_montecarlo(
                f_expr,
                float(a),
                float(b),
                int(n),
                float(confianza),
                seed,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success(f"Integral estimada: {resultado.estimacion:.12f}")
        st.write(
            f"IC {resultado.confianza*100:.1f}%: "
            f"[{resultado.ic_bajo:.12f}, {resultado.ic_alto:.12f}]"
        )
        st.write(
            f"Desvio muestral: {resultado.desvio_muestral:.6e} | "
            f"Error estandar: {resultado.error_estandar:.6e}"
        )

        # Histograma de valores transformados area * f(x_i)
        if resultado.muestras_transformadas:
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=resultado.muestras_transformadas,
                    nbinsx=40,
                    name="muestras transformadas",
                )
            )
            fig_hist.update_layout(
                title="Distribucion de muestras Monte Carlo",
                xaxis_title="(b-a)*f(X)",
                yaxis_title="Frecuencia",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Integrando en [a,b]
        try:
            xs = [float(a) + (float(b) - float(a)) * i / 300 for i in range(301)]
            ys = _eval_expr_points(f_expr, xs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)"))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(
                title="Integrando para Monte Carlo",
                xaxis_title="x",
                yaxis_title="f(x)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("No se pudo graficar la funcion en el intervalo.")


_ATAJOS_EXPRESIONES = {
    "Raices - Biseccion/Newton": [
        ("exp(x) - x**2 + 3*x - 2", "f(x) del TP"),
        ("sqrt(x) - cos(x)", "f(x) en [0,1]"),
        ("x - 2**(-x)", "f(x)=0 equivalente"),
        ("2*x*cos(x) - (x+1)**2", "funcion con trigonometria"),
    ],
    "Punto fijo (g(x))": [
        ("cos(x)", "g(x) para cos(x)-x=0"),
        ("exp(-x)", "g(x) para exp(-x)-x=0"),
        ("(x + 3/x)/2", "iteracion para aproximar sqrt(3)"),
        ("2**(-x)", "g(x)=2^{-x}"),
    ],
    "Newton - derivadas f'(x)": [
        ("3*x**2 - 2", "derivada de x**3 - 2*x - 5"),
        ("exp(x) + 2*x", "derivada de exp(x)+x**2-4"),
        ("1/x", "derivada de log(x)"),
        ("6*x**5", "derivada de x**6 - 2"),
    ],
    "RK4 y EDO": [
        ("y + t**2", "y' = y + t^2"),
        ("y*sin(t)", "y' = y sin(t)"),
        ("exp(-t) - y", "y' = e^{-t} - y"),
        ("0.1*y", "crecimiento exponencial simple"),
    ],
    "Integracion numerica": [
        ("sin(x)", "integral en [0, pi]"),
        ("exp(x)", "integral de e^x"),
        ("x**2*exp(x)", "integrando polinomio-exponencial"),
        ("1/(1+x**2)", "integrando racional"),
    ],
    "Monte Carlo": [
        ("exp(-x**2)", "integral clasica en [0,1]"),
        ("log(x)", "integral en [2,5]"),
        ("sqrt(x)", "integral en [1,4]"),
        ("sin(x)/x", "integral oscilatoria"),
    ],
}


def _panel_atajos() -> None:
    st.subheader("Chuleta de sintaxis y atajos")
    st.markdown(
        "- Potencia: `x**2` (no usar `^`)\n"
        "- Multiplicacion: `2*x` (no escribir `2x`)\n"
        "- Division: `(x+1)/(x-2)`\n"
        "- Funciones: `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`\n"
        "- Constantes: `pi`, `e`\n"
        "- Variables: usar `x`; en RK4 usar `t` y `y`"
    )
    st.info(
        "Atajo angular: podes elegir **Radianes** o **Grados** desde la barra lateral. "
        "Afecta sin/cos/tan y sus inversas en toda la app."
    )

    categoria = st.selectbox("Categoria", list(_ATAJOS_EXPRESIONES.keys()))
    opciones = _ATAJOS_EXPRESIONES[categoria]
    labels = [f"{i + 1}) {desc}" for i, (_, desc) in enumerate(opciones)]
    idx = st.selectbox("Atajo rapido", range(len(labels)), format_func=lambda i: labels[i])

    expr, desc = opciones[idx]
    st.write(f"**Seleccionado:** {desc}")
    st.text_input("Expresion lista para copiar", value=expr, key=f"atajo_{categoria}_{idx}")
    st.code(expr, language="python")

    with st.expander("Mas ejemplos utiles"):
        ejemplos = [
            ("x*cos(x) - 2*x**2 + 3*x - 1", "Biseccion/Newton"),
            ("(x+1)**(1/3)", "Punto fijo para x**3-x-1=0"),
            ("(x + 2/x)/2", "Punto fijo para aproximar sqrt(2)"),
            ("x**4 - 2*x**3 - 4*x**2 + 4*x + 4", "Polinomio de grado 4"),
            ("log(x+1)", "Natural log con desplazamiento"),
        ]
        for expr_ex, contexto in ejemplos:
            st.markdown(f"- {contexto}")
            st.code(expr_ex, language="python")


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
            "Monte Carlo",
            "Chuleta / Atajos",
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
    elif opcion == "Monte Carlo":
        _panel_montecarlo()
    elif opcion == "Chuleta / Atajos":
        _panel_atajos()
    else:
        _panel_rk4()


if __name__ == "__main__":
    main()
