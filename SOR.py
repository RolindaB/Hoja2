import numpy as np
import math
from colorama import init, Fore, Style

# Inicializar colorama
init(autoreset=True)

def get_predefined_matrix_and_vector():
    """
    Devuelve una matriz y un vector predeterminados.
    :return: Una tupla (A, b) donde A es la matriz y b es el vector.
    """
    # Definir la matriz A
    A = np.array([
        [4, -1, 0, 0, 0, 0],
        [-1, 4, -1, 0, 0, 0],
        [0, -1, 4, 0, 0, 0],
        [0, 0, 0, 4, -1, 0],
        [0, 0, 0, -1, 4, -1],
        [0, 0, 0, 0, -1, 4]
      
    ], dtype=float)
    
    # Definir el vector b (ejemplo, puedes ajustar los valores según necesites)
    b = np.array([0, 5, 0, 6, -2, 6], dtype=float)
    
    return A, b

def sor(A, b, x0, omega=1.1, tol=1e-3, max_iter=200):
    """
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método de SOR (Relaxación Sucesiva de Orden Superior).
    :param A: Matriz de coeficientes.
    :param b: Vector de constantes.
    :param x0: Valor inicial.
    :param omega: Parámetro de relajación.
    :param tol: Tolerancia para la convergencia.
    :param max_iter: Número máximo de iteraciones.
    :return: La solución x.
    """
    n = len(b)
    x = np.copy(x0)
    
    print(f"{Fore.CYAN}Iteraciones del Método SOR (ω = {omega}){Style.RESET_ALL}")
    print("=" * 50)
    
    for it in range(max_iter):
        x_old = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - sum1 - sum2) / A[i, i]
        
        # Mostrar la iteración actual
        print(f"Iteración {it + 1}: {Fore.YELLOW}{x}{Style.RESET_ALL}")
        
        # Comprobar la convergencia
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"{Fore.CYAN}Convergencia alcanzada después de {it + 1} iteraciones.{Style.RESET_ALL}")
            return x
    
    raise ValueError("El método SOR no converge")

def main():
    # Obtener la matriz y el vector predeterminados
    A, b = get_predefined_matrix_and_vector()
    
    # Valor inicial
    x0 = np.zeros(len(b))  # Valor inicial
    
    # Parámetro de relajación
    omega = 1.3
    
    # Resolver el sistema usando el método de SOR
    try:
        sol = sor(A, b, x0, omega)
        # Mostrar la solución en color verde y con formato
        print(f"\n{Fore.GREEN}{Style.BRIGHT}La solución es:{Style.RESET_ALL} {sol}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
