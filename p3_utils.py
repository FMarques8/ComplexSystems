import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from math import erf # error function for theoretical survival probability

sns.set_style("darkgrid")
palette = sns.color_palette()

## Part 1 - Two-dimensional Random Walks
# Task 1.1

def jump() -> tuple:
    val = np.random.randint(1,5)
    
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
    
    return x_data, y_data

def task1_1(n: int) -> None:
    "Solution of task 1.1."

    for _ in range(n):
        trajectory = random_walk_2d(1000)
        sns.lineplot(x=trajectory[0], y=trajectory[1], estimator=None)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('2D Random walks')
    plt.legend(['RW 1', 'RW 2', 'RW 3'])
    plt.show()

## Task 1.2

def task1_2(T: list[int], N: int = 1000) -> None:
    """Solution of task 1.2."""
    
    # Sample probability - variable initialization
    xmin = xmax = 0
    ymin = ymax = 0
    sample_probability = {}
    tmp_prob = []
    
    for t in T:
        trajectories = []
        
        # Generate trajectories
        for _ in range(N):
            trajectories.append(random_walk_2d(t))
        # we need to get a range of values of x to verify
        # can just look at last index value for each random walk and get minimum and maximum from it
        finish_site = [(x_path[-1], y_path[-1]) for x_path, y_path in trajectories]
        
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
    
    # variance
    variance = np.sum([(key[0]**2 + key[1]**2) * probability for key, probability in sample_probability.items()])
    
    print(f"""Probability condition: {prob_cond}\n
        Mean 0: {mu_x} = {mu_y}\n
        Variance {np.mean(T)}: {variance}
        """)
    
    x = []
    y = []
    
    print(sample_probability)
    for key in sample_probability.keys():
        x.append(key[0])
        y.append(key[1])
    
    # sample and theory 3d plot
    seaborn_plot = plt.axes(projection = '3d')

    seaborn_plot.plot3D(x, y, list(sample_probability.values()), label = 'Sample result', alpha = 0.4, linewidth=.5)
    seaborn_plot.plot3D(x, y, list(theory.values()), label = 'Theoretical result', color='orange', alpha = 0.7, linewidth=1)
    plt.grid()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    seaborn_plot.set_zlabel(r"$\bar{P}(x,t)$")
    plt.show()

# Part 2
## Task 2.1

def first_passage_time(boundary: int) -> int:
    """Find first passage time of 2d Random Walk with an absorving boundary."""
    
    # Initialize variables
    x = 0
    y = 0
    t = 0
    
    # Simulate random walk until it hits absorving boundary
    while x > boundary:
        tmp = jump()
        x += tmp[0]
        y += tmp[1]
        t += 1
        
        # maximum number of jumps
        if t == 50000:
            break
    return t

def get_t_binned(fpt_list: list, bin_w: int) -> list[list]:
    """Bins the t domain for task 2.1."""

    # binning process
    binned_t = [[fpt_list[0]]] # create list of bins with first fpt

    for fpt in fpt_list[1:]:
        for idx, bin in enumerate(binned_t):
            if fpt in range(bin[0], bin[0] + bin_w + 1): # delta_t + 1 to include upper limit
                binned_t[idx].append(fpt)
                break # break the loop since if it is in one bin, it cannot be in another, and this also
                    # makes it so that the else statement below is used
        else:
            binned_t.append([fpt]) # create new bin at end of list with t
        binned_t.sort() # sort list
    return binned_t

def task2_1(N: int) -> None:
    "Solution of task 2.1."
    
    # constants
    x_0 = 0
    x_b = -30
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

def levy_flight(mu: int, l_max: int, jumps: int) -> list[list[int]]:
    """Generate levy flights."""
    
    trajectory = [[0], [0]] # starting position
    
    l_r = lambda r: (1 - r * (1- l_max**(1 - mu)))**(1 / (1 - mu)) # l(r) function
    
    for _ in range(jumps):
        r = np.random.uniform()
        l = l_r(r)
        phi = 2 * np.pi * np.random.uniform()
        old_x = trajectory[0][-1]
        old_y = trajectory[1][-1] # gets last position → n-1
        new_x = old_x + l * np.cos(phi)
        new_y = old_y + l * np.sin(phi)
        
        trajectory[0].append(new_x)
        trajectory[1].append(new_y)
    
    return trajectory

def isotropic_random_walk(l: int, jumps: int) -> list[list[int]]:
    "Generate isotropic 2d random walk with fixed jump length."
    trajectory = [[0], [0]] # starting position
    
    for _ in range(jumps):
        phi = 2 * np.pi * np.random.uniform()
        old_x = trajectory[0][-1]
        old_y = trajectory[1][-1] # gets last position → n-1
        new_x = old_x + l * np.cos(phi)
        new_y = old_y + l * np.sin(phi)
        
        trajectory[0].append(new_x)
        trajectory[1].append(new_y)
    
    return trajectory

def task3_1() -> None:
    """Compare Levy flights (variable jump length) with isotropic random walks (fixed jump length)"""
    
    mu_list = [1.6, 2, 2.6]
    l_max = 1000
    N = 1000
    
    # generate isotropic 2d random walk
    isotropic_rw = isotropic_random_walk(1, N)
    
    # plot random walk and levy flights
    plt.figure(figsize=(13, 8))
    for mu in mu_list:
        levy_f = levy_flight(mu, l_max, N)
        sns.lineplot(x = levy_f[0], y = levy_f[1], marker = 'o', linestyle = 'dashed', label = f'Levy flight, $\mu={mu}$', alpha = 0.6)
    sns.lineplot(y = isotropic_rw[0], x = isotropic_rw[1], marker = 'o', linestyle = 'dashed', label = 'Isotropic random walk')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Lévy flights with varying $\mu$')
    plt.legend()
    plt.show()

def task3_2(mu_list: list[float]) -> None:
    """Function for numerical solution to task 3.2."""
    
    l_max = 1000
    
    for mu in mu_list:        
        # normalization constant
        C = (mu - 1) / (1 - l_max**(1 - mu))
        
        # generate sample set
        x = np.random.uniform(0, 1, size=100000)
        
        # lambda functions
        l_x = lambda x: (1 - x * (1- l_max**(1 - mu)))**(1 / (1 - mu)) # l(r) function
        P_l = lambda l: C * l ** (-mu) # levy flights distribution
        
        # get random numbers
        l = l_x(x)
        P = P_l(l)
        
        # plot data with histogram for distribution and the levy distribution as line plot
        sns.histplot(l, bins=1000, stat='probability', label=f'$l(x), \mu={mu}$')
        sns.lineplot(x=l, y=P, label=f'$P(l), \mu={mu}$')
        plt.legend()
    
    plt.title(f'Random numbers distribution and Levy flights distribution')
    plt.xlabel('Jump length, $l$')
    plt.ylabel('Probability')
    plt.show()