# Modelado-Simulacion

Pequena aplicacion en Python para simular y resolver modelos matematicos desde consola.

## Metodos incluidos

### 1) Busqueda de raices (equilibrios)
- **Biseccion** (con chequeo de cambio de signo en \[a,b\]).
- **Punto Fijo** \(x = g(x)\).
- **Newton-Raphson** (requiere `f(x)` y `f'(x)`).

### 2) Modelado de datos discretos
- **Interpolacion de Lagrange**.
- **Derivacion numerica por diferencia central**.

### 3) Aceleracion de convergencia
- **Delta-Cuadrado de Aitken** (usa los ultimos 3 terminos de una secuencia).

### 4) Simulacion de evolucion (EDO)
- **Runge-Kutta de orden 4 (RK4)** para `y' = f(t, y)`.

### 5) Modelo discreto adicional
- **Crecimiento logistico**:
  `x_(n+1) = r * x_n * (1 - x_n / K)`.

### 6) Integracion numerica (Newton-Cotes y Gauss)
- **Trapecio compuesto**
- **Simpson 1/3 compuesto**
- **Simpson 3/8 compuesto**
- **Rectangulo punto medio compuesto**
- **Cuadratura de Gauss-Legendre** (2 a 5 puntos)

## Requisitos

- Python 3.10 o superior

## Ejecucion

Desde la raiz del proyecto:

```bash
python3 app.py
```

## Version web interactiva (con graficos)

Tambien podes usar una interfaz web con graficos interactivos (HTML) usando Streamlit + Plotly:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run web_app.py
```

La web incluye:
- graficos de convergencia por iteracion (Biseccion, Punto Fijo, Newton, Aitken),
- curva de la funcion en intervalos configurables,
- interpolacion de Lagrange con nodos + curva interpolante,
- trayectoria RK4 y modelo logistico,
- comparacion visual de metodos de integracion numerica,
- panel de **Integracion Monte Carlo** con estimacion e intervalo de confianza,
- pagina "Chuleta / Atajos" con expresiones listas para copiar y pegar,
- selector global de modo angular (**Radianes** o **Grados**) para funciones trigonometricas.

## Interfaz web interactiva (Streamlit)

Tambien podes usar una interfaz web con formularios, tablas y seleccion de ejercicios:

```bash
python3 -m pip install -r requirements.txt
streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0
```

Luego abri en el navegador:

`http://localhost:8501`

## Estructura

- `app.py`: interfaz de consola y menu interactivo
- `modelos.py`: implementacion del motor numerico
- `web_app.py`: interfaz web interactiva con Streamlit
- `requirements.txt`: dependencias para la app web

## Modo "Ejercicios del PDF"

La app incluye una opcion de menu para resolver ejercicios tipo del trabajo practico
con entradas ya preparadas.

Incluye presets para:
- Biseccion (varios ejercicios con funcion e intervalo cargados).
- Punto fijo (casos con g(x) y x0 sugeridos).
- Newton-Raphson (f, f' y semilla).
- Aitken (aplicado a secuencias de punto fijo).
- Lagrange (nodos y punto a evaluar).
- Diferencias centrales (funcion, punto y h).
- RK4 para EDOs de valor inicial.
- Integracion numerica para ejercicios de Newton-Cotes y Gauss.

Esto permite elegir un ejercicio y cambiar solo tolerancia/iteraciones cuando haga falta.

## Notas de uso

- Las expresiones matematicas admiten funciones comunes:
  `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, etc.
- Variables esperadas por metodo:
  - Raices e interpolacion: `x`
  - RK4: `t` y `y`
- En Lagrange, los valores de `x` de los puntos deben ser distintos.
