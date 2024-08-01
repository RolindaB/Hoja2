import numpy as np

# Definir las matrices
A_2_3 = np.array([
    [4, -1, 0, 0, 0, 0],
    [-1, 4, -1, 0, 0, 0],
    [0, -1, 4, 0, 0, 0],
    [0, 0, 0, 4, -1, 0],
    [0, 0, 0, -1, 4, -1],
    [0, 0, 0, 0, -1, 4]
])

A_2_2 = np.array([
    [4, 1, -1, 1],
    [1, 4, -1, -1],
    [-1, -1, 5, 1],
    [1, -1, 1, 3]
])

A_2_1 = np.array([
    [4, 1, 1, 0, 1],
    [-1, -3, 1, 1, 0],
    [2, 1, 5, -1, -1],
    [-1, -1, -1, 4, 0],
    [0, 2, -1, 1, 4]
])

# Función para verificar si una matriz es simétrica
def es_simetrica(A):
    return np.allclose(A, A.T)

# Función para verificar si una matriz es positivo-definida
def es_positivo_definida(A):
    return np.all(np.linalg.eigvals(A) > 0)

# Verificar las propiedades de las matrices
sistemas = {
    "Sistema 2.3": A_2_3,
    "Sistema 2.2": A_2_2,
    "Sistema 2.1": A_2_1
}

resultados = {}

for nombre, A in sistemas.items():
    simetrica = es_simetrica(A)
    positivo_definida = es_positivo_definida(A)
    resultados[nombre] = {
        "Simétrica": simetrica,
        "Positivo-definida": positivo_definida
    }

# Mostrar los resultados
for nombre, resultado in resultados.items():
    print(f"{nombre}:")
    print(f"  Simétrica: {resultado['Simétrica']}")
    print(f"  Positivo-definida: {resultado['Positivo-definida']}")
    print()