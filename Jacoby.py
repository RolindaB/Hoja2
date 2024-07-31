import numpy as np
import math
from colorama import init, Fore, Style

# Inicializar colorama
init(autoreset=True)

def evaluate_expression(expr):
    """
    Evalúa una expresión matemática dada como una cadena.
    :param expr: Expresión matemática en forma de cadena.
    :return: Valor evaluado de la expresión.
    """
    try:
        return eval(expr, {"math": math})
    except Exception as e:
        print(f"Error al evaluar la expresión: {e}")
        return None

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
        [0,-0, 0, 4, -1, 0],
        [0,-0, 0, -1, 4, -1],
        [0,-0, 0, 0, -1, 4]
        
       
    ], dtype=float)
    
    # Definir el vector b (ejemplo, puedes ajustar los valores según necesites)
    b = np.array([0, 5, 0, 6, -2, 6], dtype=float)
    
    return A, b

def jacobi(A, b, x0, tol=1e-3, max_iter=100):
    """
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método de Jacobi.
    :param A: Matriz de coeficientes.
    :param b: Vector de constantes.
    :param x0: Valor inicial.
    :param tol: Tolerancia para la convergencia.
    :param max_iter: Número máximo de iteraciones.
    :return: La solución x.
    """
    n = len(b)
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    
    print(f"{Fore.CYAN}Iteraciones del Método de Jacobi{Style.RESET_ALL}")
    print("=" * 50)
    
    for it in range(max_iter):
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Mostrar la iteración actual
        print(f"Iteración {it + 1}: {Fore.YELLOW}{x_new}{Style.RESET_ALL}")
        
        # Comprobar la convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"{Fore.CYAN}Convergencia alcanzada después de {it + 1} iteraciones.{Style.RESET_ALL}")
            return x_new
        
        x = np.copy(x_new)
    
    raise ValueError("El método de Jacobi no converge")

def main():
    # Obtener la matriz y el vector predeterminados
    A, b = get_predefined_matrix_and_vector()
    
    # Valor inicial
    x0 = np.zeros(len(b))  # Valor inicial
    
    # Resolver el sistema usando el método de Jacobi
    try:
        sol = jacobi(A, b, x0)
        # Mostrar la solución en color verde y con formato
        print(f"\n{Fore.GREEN}{Style.BRIGHT}La solución es:{Style.RESET_ALL} {sol}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
