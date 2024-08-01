import numpy as np

def calcular_radio_espectral(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    T_GS = np.dot(np.linalg.inv(D + L), -U)
    return max(abs(np.linalg.eigvals(T_GS)))

def calcular_omega_optimo(A):
    rho = calcular_radio_espectral(A)
    return 2 / (1 + np.sqrt(1 - rho**2))

def sor_method(A, b, X0, omega, tolerance=1e-3, max_iterations=100):
    n = len(b)
    X = X0.copy()
    for k in range(max_iterations):
        X_new = X.copy()
        for i in range(n):
            s1 = sum(A[i][j] * X_new[j] for j in range(i))
            s2 = sum(A[i][j] * X[j] for j in range(i + 1, n))
            X_new[i] = (1 - omega) * X[i] + (omega * (b[i] - s1 - s2)) / A[i][i]
        if np.linalg.norm(X_new - X, ord=np.inf) < tolerance:
            return X_new, k + 1
        X = X_new
    return X, max_iterations

def main():
    # Definimos las matrices de coeficientes y los vectores de términos independientes
    sistemas = [
        (np.array([
            [4, 1, -1, 1],
            [1, 4, -1, -1],
            [-1, -1, 5, 1],
            [1, -1, 1, 3]
        ]), np.array([-2, -1, 0, 1])),
        (np.array([
            [4, -1, 0, 0, 0, 0],
            [-1, 4, -1, 0, 0, 0],
            [0, -1, 4, 0, 0, 0],
            [0, 0, 0, 4, -1, 0],
            [0, 0, 0, -1, 4, -1],
            [0, 0, 0, 0, -1, 4]
        ]), np.array([0, 5, 0, 6, -2, 6]))
    ]

    for idx, (A, b) in enumerate(sistemas, start=1):
        # Calcular omega óptimo
        omega_opt = calcular_omega_optimo(A)
        print(f"\nOmega óptimo para el sistema {idx}: {omega_opt:.4f}")

        # Definir el vector inicial
        X0 = np.zeros(len(b))

        # Resolver el sistema usando el método de SOR
        solution, iterations = sor_method(A, b, X0, omega_opt)
        print_solution(f"{idx}", solution, iterations)

def print_solution(system_label, solution, iterations):
    print(f"\nSistema {system_label}:")
    for i, x in enumerate(solution):
        print(f"  x{i + 1} = {x:.6f}")
    print(f"  Número de iteraciones: {iterations}")

if __name__ == "__main__":
    main()