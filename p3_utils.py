import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from random import randint
from math import erf # error function for theoretical survival probability

sns.set_style("darkgrid")
palette = sns.color_palette()

## Part 1 - Two-dimensional Random Walks
# Task 1.1

def jump() -> tuple:
    val = randint(1,4)
    
    if val == 1:
        return (0, 1)
    elif val == 2:
        return (1, 0)
    elif val == 3:
        return (0, -1)
    else:
        return (-1, 0)

def random_walk_2d(end_time: int) -> tuple[list[int], list[int]]:
    """Simulates 2D random walk"""

    # Initialize variables
    x, y = (0, 0)
    x_data = [x]
    y_data = [y]
    
    # Simulate random walk
    for t in range(end_time):
        tmp = jump()
        x += tmp[0]
        y += tmp[1]
        x_data.append(x)
        y_data.append(y)
    
    return list(zip(x_data, y_data))

def task1_1(n: int) -> None:
    "Simulate and plot n 2D random walks."

    for _ in range(n):
        trajectory = random_walk_2d(100)
        plt.plot(trajectory)
    plt.xlabel('Position $x$')
    plt.ylabel('Position $y$')
    plt.title('2D Random walks')
    plt.grid()
    plt.legend(['RW 1', 'RW 2', 'RW 3'])
    plt.show()

## Task 1.2

def task1_2(T: list[list[int]] = [[50000, 50001]], N: int = 1000) -> None:
    """Simulate n random walks and compute probability of
    finding a particle at site x and time tf
    
    Args:
        T - final time to look for the particle
        n - number of random walks to compute (samples)
    """
    
    for tf in T:
        
        # Sample probability - variable initialization
        xmin = xmax = 0
        ymin = ymax = 0
        sample_probability = {}
        tmp_prob = []
        
        for t in tf:
            trajectories = []
            
            # Generate trajectories
            for _ in range(N):
                trajectories.append(random_walk_2d(t))
            # we need to get a range of values of x to verify
            # can just look at last index value for each random walk and get minimum and maximum from it
            finish_site = [trajectory[-1] for trajectory in trajectories]
            
            # interval for finish site check
            for finish in finish_site:
                xmin = min(xmin, finish[0])
                ymin = min(ymin, finish[1])
                
                xmax = max(xmax, finish[0])
                ymax = max(ymax, finish[1])
            
            occurrences = {(x, y): 0 for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)} # initialization
            
            for key in occurrences:
                occurrences[key] = sum([int(finish == key) for finish in finish_site]) # count number of occurrences of each finish site
            
            # print(occurrences)
            tmp_prob.append({key: val/N for key, val in occurrences.items()})
            
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                sample_probability[(x, y)] = (tmp_prob[0].get((x, y), 0) + tmp_prob[1].get((x, y), 0)) / 2 # sample average between the odd and even random walks

        # # Theoretical probability
        P = lambda x, y, t: np.exp(-(x**2 + y**2) / t) / (t * np.pi)  # define a lambda function P(x,t)
        theory = {(x, y): P(x, y, t) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)}
        
        # normalization conditions
        prob_cond = np.sum(list(sample_probability.values()))
        
        # mean values
        mu_x = np.sum(list(map(lambda x, prob: x * prob, range(xmin, xmax + 1), sample_probability.values())))
        mu_y = np.sum(list(map(lambda y, prob: y * prob, range(ymin, ymax + 1), sample_probability.values())))
        
        variance = np.sum(list(map(lambda x, y, prob: (x**2 * prob + y**2 * prob), range(xmin, xmax + 1), range(ymin, ymax + 1), sample_probability.values())))
        
        print(f"""Probability condition: {prob_cond}\n
            Mean 0: {mu_x} = {mu_y}\n
            Variance {np.mean(tf)}: {variance}
            """)
        
        x = []
        y = []
        for key in sample_probability.keys():
            x.append(key[0])
            y.append(key[1])
        
        # sample and theory 3d plot
        seaborn_plot = plt.axes(projection = '3d')
        seaborn_plot.scatter3D(x, y, sample_probability.values(), label = 'Sample result', alpha = 1)
        seaborn_plot.scatter3D(x, y, theory.values(), label = 'Theoretical result', color='orange', alpha = 0.1)
        plt.grid()
        plt.xlabel(r"Position $x$")
        plt.ylabel(r"Position $y$")
        seaborn_plot.set_zlabel(r"Probability $\bar{P}(x,t)$")
        plt.show()

# Part 2
## Task 2.1

def first_passage_time(boundary: int):
    
    # Initialize variables
    x = 0
    y = 0
    t = 0
    
    # Simulate random walk
    while x > boundary:
        tmp = jump()
        x += tmp[0]
        y += tmp[1]
        t += 1
        
        # maximum number of jumps
        if t == 50000:
            break
    return t

def get_t_binned(fpt_list: list, bin_w: int):
    # binning process
    binned_t = [[fpt_list[0]]] # create list of bins with first fpt

    for fpt in fpt_list[1:]:
        for idx, bin in enumerate(binned_t):
            # print(bin)
            if fpt in range(bin[0], bin[0] + bin_w + 1): # delta_t + 1 to include upper limit
                binned_t[idx].append(fpt)
                break # break the loop since if it is in one bin, it cannot be in another, and this also
                    # makes it so that the else statement below is used
        else:
            binned_t.append([fpt]) # create new bin at end of list with t
        binned_t.sort() # sort list
    return binned_t

def task2_1(N: int):
    
    # constants
    x_0 = 0
    x_b = -3
    D = 1/4
    
    # Simulation results
    fpt_list = sorted([first_passage_time(x_b) for _ in range(N)])
    binned_t = get_t_binned(fpt_list, 10)
    
    N_fpt = {bin[0]: len(bin) for bin in binned_t}

    fpt_distribution = {key: counts / (N * 10) for key, counts in N_fpt.items()}
    
    # use the binned time domain to increase the efficiency by not needing to read every element in every list
    def survival_probability(t: int): 
        "Compute survival probability for each t"
        # filter out bins which t+delta_t is greater than t, which means that all of the 
        # particles didnt survive, including at time t, since they could get trapped at time t
        # N_survived = len(list(filter(lambda x: x > t, fpt_list)))
        N_survived = N
        for fpt in fpt_list:
            if fpt < t:
                N_survived -= 1
        return N_survived / N
    
    S_t = {t: survival_probability(t) for t in range(1, 50001)}
    
    # Theoretical results
    # not used be
    F = lambda t: abs(x_b - x_0) * np.exp(- ((x_b - x_0)**2 / (4 * D * t))) / (2 * np.sqrt(np.pi * D * t**3))
    F_approx = lambda t: abs(x_b - x_0) / (2 * np.sqrt(np.pi * D * t**3))
    
    S = lambda t: erf(abs(x_b - x_0) / (2 * np.sqrt(D * t)))
    S_approx = lambda t: abs(x_b - x_0) / np.sqrt(np.pi * D * t)
    
    # Plots
    x = range(1, 50001) # values to plot the theoretical results

    sns.lineplot(x = x, y = list(map(F, x)), c=palette[0], label = 'Exact theoretical result')
    sns.lineplot(x = x, y = list(map(F_approx, x)), linestyle = 'dashed', c = palette[0], label = 'Approximate theoretical result')
    sns.lineplot(x = list(fpt_distribution.keys()), y = list(fpt_distribution.values()), c = palette[1], label = 'Simulation result') 
    plt.xscale('log') # put axis on logarithmic scale
    plt.yscale('log')
    plt.title('First passage time probability density at large $t$')
    plt.xlabel('$\ln{t}$')
    plt.ylabel('$\ln{F(t)}$')
    plt.show()
    
    sns.lineplot(x=x, y=map(S, x), c=palette[0], label = 'Exact theoretical result')
    sns.lineplot(x=x, y=map(S_approx, x), linestyle='dashed', c=palette[0], label = 'Approximate theoretical result')
    sns.lineplot(S_t, c=palette[1], label = 'Simulation result')
    plt.xlabel('$t$')
    plt.ylabel('Survival probability $S(t)$')
    plt.title('Survival probability for each $t$')
    plt.show()

# Part 3
## Task 3.1