import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Crear el DataFrame con los datos de Benetton
data = {
    'Advertising': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Sales': [2, 4, 6, 8, 10, 12, 14, 16, 18]
}

df = pd.DataFrame(data)

# Calcular los valores necesarios para la regresión
n = len(df)
sum_x = df['Advertising'].sum()
sum_y = df['Sales'].sum()
sum_xy = (df['Advertising'] * df['Sales']).sum()
sum_x_squared = (df['Advertising'] ** 2).sum()

# Calcular β₁ (pendiente)
beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)

# Calcular β₀ (intersección)
beta_0 = (sum_x_squared * sum_y - sum_x * sum_xy) / (n * sum_x_squared - sum_x ** 2)

# Crear visualización
plt.figure(figsize=(10, 6))
plt.scatter(df['Advertising'], df['Sales'], color='blue', label='Datos reales')
plt.plot(df['Advertising'], beta_0 + beta_1 * df['Advertising'], color='red', label='Línea de regresión')
plt.xlabel('Advertising')
plt.ylabel('Sales')
plt.title('Regresión Lineal: Ventas vs Publicidad')
plt.legend()
plt.grid(True)

# Imprimir resultados
print("\nResultados del análisis de regresión lineal:")
print(f"β₀ (Intersección) = {beta_0:.4f}")
print(f"β₁ (Pendiente) = {beta_1:.4f}")
print("\nEcuación de la línea de regresión:")
print(f"y = {beta_0:.4f} + {beta_1:.4f}x")

# Crear tabla de resultados
print("\nTabla de cálculos:")
df['x²'] = df['Advertising'] ** 2
df['xy'] = df['Advertising'] * df['Sales']
df['y_pred'] = beta_0 + beta_1 * df['Advertising']
print(df)

# Imprimir sumas totales
print("\nSumas totales:")
print(f"Σx = {sum_x}")
print(f"Σy = {sum_y}")
print(f"Σxy = {sum_xy}")
print(f"Σx² = {sum_x_squared}")