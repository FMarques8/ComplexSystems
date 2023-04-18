# Functions for project 2 of Modelling of Complex Systems
# March 2023 Francisco Marques

####
# Modules

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import random

sns.color_palette("Set2")

####
# General functions

def jump(p: float = 0.5) -> int:
    """Simulates a jump, either to the right (1) or left (-1)
    Args
        p - probability of jumping to the left
    """
    if random() > p:
        return 1
    else:
        return -1

def random_walk(start_position: int = 0, start_time: int = 0, end_time: int = 50) -> list[int]:
    """Simulates 1D random walk"""
    
    # Initialization variables
    x = start_position
    trajectory = [x]
    
    # Do random walk
    for t in range(start_time, end_time):
        x += jump(0.5) # make the n particles jump
        trajectory.append(x) 
    
    return trajectory

def random_walk_with_drift(start_position: int = 0, start_time: int = 0, end_time: int = 50, delta: float = 0.015) -> list[int]:
    """Simulates 1D random walk"""
    
    # Initialization variables
    x = start_position
    trajectory = [x]
    
    # Do random walk
    for t in range(start_time, end_time):
        x += jump(0.5 - delta) # make the n particles jump
        trajectory.append(x) 
    
    return trajectory

## Part 1
# Task 1.1

def task1_1(n: int = 3) -> None:
    """Simulates and plots n 1D random walks."""
    
    trajectory = []
    
    for _ in range(n):
        trajectory.append(random_walk(0, 0, 50))
            
    # Graphical results
    sns.lineplot(trajectory)
    plt.title("Random walk with $p=q=0.5$")
    plt.ylabel("Position $x$")
    plt.xlabel("Time $t$")
    plt.grid(True)
    plt.show()

def task1_2(T: list[int, int] = [[40, 41], [400, 401], [4000, 4001]], N: int = 50000) -> None:
    """Simulate n random walks and compute probability of
    finding a particle at site x and time tf
    
    Args:
        T - final time to look for the particle
        n - number of random walks to compute (samples)
    """
    
    for tf in T:
        
        # Sample probability - variable initialization
        xmin = xmax = 0
        sample_probability = {}
        tmp_prob = []
        
        for t in tf:
            
            trajectory = []
            
            # Generate trajectories
            for _ in range(N):
                trajectory.append(random_walk(0, 0, t))
            
            # we need to get a range of values of x to verify
            # can just look at last index value for each random walk and get minimum and maximum from it
            finish_site = [x[-1] for x in trajectory]
            
            # interval for finish site check
            if np.min(finish_site) < xmin: 
                xmin = np.min(finish_site)
            if np.max(finish_site) > xmax: 
                xmax = np.max(finish_site)
            
            occurrences = {i: None for i in range(xmin, xmax + 1)} # initialization
            
            for x in occurrences:
                occurrences[x] = sum([int(final == x) for final in finish_site]) # count number of occurrences of each finish site
            
            tmp_prob.append({x: y/N for x, y in occurrences.items()})
        
        for key in range(xmin, xmax + 1):
            sample_probability[key] = (tmp_prob[0].get(key, 0) + tmp_prob[1].get(key, 0)) / 2 # sample average between the odd and even random walks
                
        # # Theoretical probability
        P = lambda x, t: np.exp(-(x**2) / (2 * t)) / np.sqrt(2 * t * np.pi)  # define a lambda function P(x,t)
        theory = {x: P(x, t) for x in range(xmin, xmax + 1)}
        
        # normalization conditions
        prob_cond = np.sum(list(sample_probability.values()))
        mu = np.sum(list(map(lambda position, p: position * p, range(xmin, xmax + 1), sample_probability.values())))
        variance = np.sum(list(map(lambda position, p: (position - mu)**2 * p, range(xmin, xmax + 1), sample_probability.values())))
        
        print(f"""Probability condition: {prob_cond}\n
            Mean 0: {mu}\n
            Variance {np.mean(tf)}: {variance}
            """)
        
        sns.scatterplot(sample_probability, label = 'Sample result', s = 12)
        sns.lineplot(theory, label = 'Theoretical result') # do i have to plot for t and t+1, or the average value??? ASK THIS!!!!!!!!!!!!!!
    plt.grid()
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Probability $\bar{P}(x,t)$")
    plt.show()

# Part 2

def task2(T: list[int, int] = [[40, 41], [400, 401], [4000, 4001]], N: int = 50000, delta: float = 0.015) -> None:
    """Simulate n random walks and compute probability of
    finding a particle at site x and time tf
    
    Args:
        T - final time to look for the particle
        n - number of random walks to compute (samples)
        delta - drift constant
    """
    
    for tf in T:
        
        # Sample probability - variable initialization
        xmin = xmax = 0
        sample_probability = {}
        tmp_prob = []
        
        for t in tf:
            
            trajectory = []
            
            # Generate trajectories
            for _ in range(N):
                trajectory.append(random_walk_with_drift(0, 0, t, delta))
            
            # we need to get a range of values of x to verify
            # can just look at last index value for each random walk and get minimum and maximum from it
            finish_site = [x[-1] for x in trajectory]
            
            # interval for finish site check
            if np.min(finish_site) < xmin: 
                xmin = np.min(finish_site)
            if np.max(finish_site) > xmax: 
                xmax = np.max(finish_site)
            
            occurrences = {i: None for i in range(xmin, xmax + 1)} # initialization
            
            for x in occurrences:
                occurrences[x] = sum([int(final == x) for final in finish_site]) # count number of occurrences of each finish site
            
            tmp_prob.append({x: y/N for x, y in occurrences.items()})
        
        for key in range(xmin, xmax + 1):
            sample_probability[key] = (tmp_prob[0].get(key, 0) + tmp_prob[1].get(key, 0)) / 2 # sample average between the odd and even random walks
                
        # # Theoretical probability
        P = lambda x, t: np.exp(-(x - 2 * t * delta)**2 / (2 * t)) / np.sqrt(2 * t * np.pi)  # define a lambda function P(x,t)
        theory = {x: P(x, t) for x in range(xmin, xmax + 1)}
        
        # normalization conditions
        prob_cond = np.sum(list(sample_probability.values()))
        mu = np.sum(list(map(lambda position, p: position * p, range(xmin, xmax + 1), sample_probability.values())))
        variance = np.sum(list(map(lambda position, p: (position - mu)**2 * p, range(xmin, xmax + 1), sample_probability.values())))
        
        print(f"""Probability condition: {prob_cond}\n
            Mean {round(2 * delta * np.mean(tf), 4)}: {mu}\n
            Variance {np.mean(tf)}: {variance}
            """)
        
        sns.scatterplot(sample_probability, label = 'Sample result', s = 12)
        sns.lineplot(theory, label = 'Theoretical result') # do i have to plot for t and t+1, or the average value??? ASK THIS!!!!!!!!!!!!!!
    plt.grid()
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Probability $\bar{P}(x,t)$")
    plt.show()
