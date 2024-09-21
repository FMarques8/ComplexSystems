# Functions for project 4 of Modelling of Complex Systems
# May 2023 Francisco Marques

####
# Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

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
    return np.sinh(H / T) / np.sqrt(np.sinh(H / T)**2 + np.exp((-4 * 1) / T))

# EX 1

# m function of T
T = np.arange(0.1, 10.01, step = 0.05)
m_sim = []
m_theory = []

for temp in T:
    m_sim.append(metropolis(1000, temp))
    m_theory.append(exact_magnetization(temp))

sns.lineplot(x=T, y=m_sim, label = 'Simulation')
sns.lineplot(x=T, y=m_theory, label = 'Exact')
plt.legend()
plt.title('Mean magnetic moment vs Temperature')
plt.ylabel('$m(H=0.1,T)$')
plt.xlabel('Temperature $T$')
plt.show()

# EX 2