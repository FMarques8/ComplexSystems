
import numpy as np
from collections import Counter
from scipy.special import factorial 
import matplotlib.pyplot as plt
import seaborn as sns

def create_network(N: int, L: int):
    "Generate Erdős-Rényi network of size N (dim N x N) with L edges"
    
    A = np.zeros((N, N))

    for _ in range(L):
        while True:
            # draw uniform random nodes
            i, j = np.random.randint(0, N, 2)
            
            if i != j or A[i, j] == 1:
                break
        A[i, j] = 1
        A[j, i] = 1
    return A

def get_degree(A):
    "Returns degrees of network A"
    return np.sum(A, axis = 0)

def degree_distribution(degree_vector, N: int):
    "Returns degree distribution"
    
    degree, N_degree = np.unique(degree_vector, return_counts = True)
    P_degree = {int(x): count / N for x, count in zip(degree, N_degree)}
    return P_degree

def average_degree(N: int, L: int, m: int):
    "Returns sample average degree and averaged sample degree distribution"
    
    avg_P = Counter()
    
    for _ in range(m):
        A = create_network(N, L)
        degree = get_degree(A)
        P_degree = degree_distribution(degree, N)
        avg_P.update(P_degree)
    
    for q in avg_P:
        avg_P[q] /= m
    avg_degree = np.sum(list(map(lambda q: q * avg_P[q], avg_P.keys())))
    
    return avg_degree, dict(sorted(avg_P.items())), A, degree

def branching_coeff(avg_degree, P_degree):
    "Returns the branching coefficient based on the mean degree"
    return (1 / avg_degree) * np.sum([P_degree[degree] * degree * (degree - 1) for degree in P_degree])

def moment(N: int, q, p: int):
    "Returns p-th moment of given degree array"
    return (1 / N) * np.sum(q ** p)

def P_theory(q: int, c: int):
    "Returns theoretical degree distribution"
    return np.exp(-c) * (c ** q) / factorial(q)

def cluster_coeff(A, q):
    "Returns the clustering coefficient of a network A"
    # We can discard N since it cancels out with n_pt and n_tr
    n_pt = np.sum(q * (q - 1)) # number of possible triangles
    # sum(sum(sum(...))) can be converted to A^3
    n_tr = np.trace(np.linalg.matrix_power(A, 3))
    return n_tr / n_pt

def pearson_corr(A, q, N):
    "Returns the Pearson coefficient of a network A"

    Q =(np.sum(q ** 2)/N) / (np.sum(q)/N)
    sigma_squared = (np.sum(q ** 3) / np.sum(q)) - (np.sum(q ** 2)**2 / np.sum(q) ** 2) # same as before

    num = np.sum(np.sum(A * (q-Q), axis = 1) * (q-Q))
    den = np.sum(q) * sigma_squared
    return num / den