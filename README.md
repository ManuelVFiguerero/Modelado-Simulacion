# Modelado-Simulacion

Pequena aplicacion en Python para simular modelos matematicos desde consola.

## Modelos incluidos

1. **Interpolacion de Lagrange**
   - Carga un conjunto de puntos \((x_i, y_i)\).
   - Evalua el polinomio interpolante en un valor `x`.

2. **Metodo de Punto Fijo**
   - Resuelve iterativamente \(x = g(x)\).
   - Permite ingresar expresiones como:
     - `cos(x)`
     - `exp(-x)`
     - `(x + 2/x) / 2`
   - Muestra tabla de iteraciones y error por paso.

3. **Modelo Logistico Discreto**
   - Simula:
     \[
     x_{n+1} = r x_n \left(1 - \frac{x_n}{K}\right)
     \]
   - Reporta la evolucion de la serie en cada paso.

## Requisitos

- Python 3.10 o superior

## Ejecucion

Desde la raiz del proyecto:

```bash
python app.py
```

## Estructura

- `app.py`: interfaz de consola y flujo del menu
- `modelos.py`: implementacion de los metodos numericos

## Notas

- El evaluador de expresiones para punto fijo habilita funciones comunes de `math`:
  `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, entre otras.
- Para interpolacion de Lagrange, los valores de `x` en los puntos deben ser distintos.
