import numpy as np
import sys

def load_matrix(filename):
    with open(filename, 'r') as f:
        return np.array([[int(x) for x in line.strip().split()] for line in f])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify.py <matrix_a> <matrix_b> <result>")
        sys.exit(1)
        
    try:
        a = load_matrix(sys.argv[1])
        b = load_matrix(sys.argv[2])
        result = load_matrix(sys.argv[3])
        
        true_result = np.dot(a, b)
        comparison = np.array_equal(true_result, result)
        print("True" if comparison else "False")
    except Exception as e:
        print(f"False (Ошибка: {str(e)})")