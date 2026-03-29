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

## Requisitos

- Python 3.10 o superior

## Ejecucion

Desde la raiz del proyecto:

```bash
python3 app.py
```

## Estructura

- `app.py`: interfaz de consola y menu interactivo
- `modelos.py`: implementacion del motor numerico

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

Esto permite elegir un ejercicio y cambiar solo tolerancia/iteraciones cuando haga falta.

## Notas de uso

- Las expresiones matematicas admiten funciones comunes:
  `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, etc.
- Variables esperadas por metodo:
  - Raices e interpolacion: `x`
  - RK4: `t` y `y`
- En Lagrange, los valores de `x` de los puntos deben ser distintos.
