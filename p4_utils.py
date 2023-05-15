# Functions for project 4 of Modelling of Complex Systems
# May 2023 Francisco Marques

####
# Modules
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #progress bar

# Functions

# Part 1 - Ising Model on the ring

def E(state: np.array, n: int, J: int, H: float):
    "Energy of ising model on a given spin"

    # sigma_{N+1} equiv to sigma_1
    if n+1 == len(state):
        return -J * state[n] * (state[n - 1] + state[0]) - H * state[n]
    else:
        return -J * state[n] * (state[n - 1] + state[n + 1]) - H * state[n] 

def metropolis(N: int, T: float):
    """Metropolis algorithm"""
    
    J = 1 # coupling constant 
    H = 0.1 #fixed value   

    microstate = np.random.choice(a = [-1, 1], size = N)
    
    # select a random spin
    n = np.random.randint(0, len(microstate), size = 1)

    # temporary array with flipped spin
    new_state = microstate
    new_state[n] = -new_state[n]
    
    delta_E = E(new_state, n, J, H) - E(microstate, n, J, H)
    
    if delta_E <= 0:
        microstate[n] = -microstate[n]

    else:
        w = np.exp(-delta_E/T)
        r = np.random.uniform()
        
        if r < w:
            microstate[n] = -microstate[n]
    
    m = (1 / N) * np.sum(microstate)
    return m

def exact_magnetization(T):
    """Returns exact values of the magnetization"""
    H = 0.1
    return np.sinh(H / T) / np.sqrt(np.sinh(H / T)**2 + np.exp((-4 * J) / T))

T = np.arange(0.1, 10.01, step = 0.05)

m = []
magnetization = []

for k in tqdm(range(1, 101)):
    for t in range(k*100 + 1):
        m.append(metropolis(1000, np.random.choice(T, 1)))
    magnetization.append(np.sum(m) / (k * 100))


plt.plot(np.arange(1, 101)*100, magnetization)
plt.show()