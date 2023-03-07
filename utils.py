import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate

# Part 1 - Probability
# Task 1.2

def average_value(M: int):
    """Computes the numerical averaged value, 〈𝑥〉, of M natural numbers."""

    return np.mean(np.random.randint(1, M+1, size = M))

def variance(M: int):
    """Computed the numerical variance of M natural numbers"""

    return np.var(np.random.randint(1, M+1, size=M))

def probability_leq_x(x: int, M: int):
    """Computes the numerical probability of value being lesser or equal than x"""

    draw = np.random.randint(1, M+1, size = M) # random sample of size M
    count = sum(1 for i in draw if i <= x) # sum of values <= 60
    return count/M

def task1_2():
    """Function to run numerical test to compare with analytical results for task 1.2."""

    M = 100
    x = 60

    for N in [10**2, 10**4, 10**6]:
        print(f'{N} trials')
        
        # Analytical average
        mu = (M+1)/2
        
        # Analytical variance
        sigma = (M**2-1)/12
        
        # Analytical P(x<=60)
        prob = (x-1)/(M-1)
        
        # lists for numerial results
        average = []
        var = []
        num_prob = []
        
        for _ in range(N):
            average.append(average_value(M))
            var.append(variance(M))
            num_prob.append(probability_leq_x(x, M))
        
        print(f'Theoretical average: {round(mu, 5)}, Numerical average: {round(np.mean(np.array(average)), 5)}')
        print(f'Theoretical variance: {round(sigma, 5)}, Numerical variance: {round(np.mean(np.array(var)), 5)}')
        print(f'Theoretical probability: {round(prob, 5)}, Numerical probability: {round(np.mean(np.array(num_prob)), 5)}')
        print()

# Task 1.3.
def task1_3(N: int):
    """Function to solve the product mean value task 1.3."""
    
    print("Theoretical results")
    print("Mean of x: 0.5")
    print("Mean of y: 0.5")
    print("Mean of z: 0.25")
    
    for n_trials in [10**2, 10**4, 10**6]:
        
        x_mean = []
        y_mean = []
        z_mean = []
        
        for _ in range(n_trials):
            
            # Initialize random vectors with N random uniformly distributed number
            x = np.random.rand(N)
            y = np.random.rand(N)

            # product vector with N elements
            z = [a*b for a, b in zip(x, y)]

            # means
            x_mean.append(np.mean(x))
            y_mean.append(np.mean(y))
            z_mean.append(np.mean(z))
        
        print(f'{n_trials} trials')
        print(f'Mean of x: {round(np.mean(x_mean), 5)}') 
        print(f'Mean of y: {round(np.mean(y_mean), 5)}')
        print(f'Mean of z: {round(np.mean(z_mean), 5)}')
        print()


# Part 2 - Probability density
# Task 2.1

def task2_1():
    """Plots the probability density distribution of a random variable x in the interval [0,1]"""
    
    for N in [10**2, 10**4, 10**6]:
    
        x = np.random.rand(N)

        sns.histplot(x, binwidth = 0.005, stat = 'density', kde = True)
        plt.axhline(y=1, color = 'r') # draws line of theoretical value of the counts
        plt.show()

# Task 2.2

def task2_2():
    """Computes numerical mean, variance of uniform random variable x in interval [0,1]"""
    
    for N in [10**2, 10**4, 10**6]:
        
        x = np.random.rand(N)
        
        avg = np.mean(x)
        var = np.var(x)
        
        print(f'{N} samples')
        print(f'Mean of x: {round(avg, 5)}')
        print(f'Variance of x: {round(var, 5)}')

# Task 2.3

def task2_3():
    """Computes the probability density of a variable y =sqrt(rand(1))"""
    
    for N in [10**2, 10**4, 10**6]:
        
        y = np.sqrt(np.random.rand(N))
        
        sns.histplot(y, binwidth = 0.005, stat = 'density')
        sns.lineplot(x = np.linspace(0, 1, N), y = 2*np.linspace(0, 1, N), color = 'r')
        plt.show()

# Task 3.1

def task3():
    """Computes the probability density distribution function of a variably Y = mean(x)"""
    
    #####
    # Steps 1 to 3
    
    N = 10**4
    width = 0.005
    k = int(1/width - 1)

    for n in [10, 100, 1000]:
        
        print(f'{n} samples')
        
        Y = [[] for _ in range(k+1)] # vector of y_m; k+1 so it is in the range [0,200] and not [0, 199]
        X = []
        
        
        for _ in range(N):
            x = np.random.rand(n)
            X.append(x)
            y = np.mean(x)
            for bin in range(len(Y)):
                if y >= bin*width and y < (bin+1)*width:
                    Y[bin].append(y)

        y_dist = list(map(len, Y))

        y_density = list(map(lambda bin: bin/(N*width), y_dist))

        norm_cond = round(sum(y_density)*width, 4)
        print(f'Normalization condition value: {round(norm_cond, 5)}')
        
        ####
        # Step 4
        # Mean
        y_k = list(map(lambda bin: (bin + 0.5) * width, range(len(Y)))) # center of each bin
        mean_Y = sum(list(map(lambda center, prob: center*prob*width, y_k, y_density)))
        print(f'Mean of Y: {round(mean_Y, 5)}')
        
        # Variance
        var_Y = sum(list(map(lambda center, prob: (center - mean_Y)**2 * (prob*width), y_k, y_density)))
        print(f'Variance of Y: {round(var_Y, 5)}')
        
        ####
        # Step 5
        # Mean
        mean_x = sum(list(map(lambda x: np.sum(x)/n, X)))/N
        print(f'Mean of x: {round(mean_x, 5)}')
        
        # Variance
        var_x = sum(list(map(lambda x: np.sum((x - mean_x)**2) / n, X))) / N
        print(f'Variance of x: {round(var_x, 5)}')
        
        print(f"sigma^2/n: {round(var_x/n, 5)}, variance of Y: {round(var_Y, 5)}\n")

# Task 4.1

def throwing_balls():
    """Function for first project's part 4 - throwing balls"""
    
    M = 9
    N = 21
    K = [10**2, 10**4, 10**6]
    
    for k in K:
        print(f'{k} samples')
        n = [] # number of balls in box 3 for each trial
        
        for _ in range(k):
            x = np.random.randint(1, M + 1, N) # interval is open at high value
            
            n.append(sum(1 for ball in x if ball == 3)) # counts # of balls in box 3
        
        # Step 4
        # probability of having 1 or more ball in box 3
        
        ball_in_box3 = [int(ball > 0) for ball in n]
        print(f'Number of attempts with at least 1 ball in box 3: {sum(ball_in_box3)}')
        print(f'Probability of finding at least 1 ball in box 3: {sum(ball_in_box3)/k}')
        
        target = 3 # n balls in box 3
        n_balls_in_box3 = [int(ball == target) for ball in n]
        print(f'Probability of finding {target} balls in box 3: {sum(n_balls_in_box3)/k}')
        
        # Step 5
        p = 1/M
        mean = N*p
        var = N*p
        
        num_l = [] # number of attempts with l balls in box 3
        prob_l = [] # probability of having l balls in box 3
        
        for l in range(N+1):
            num_l.append(sum([int(ball == l) for ball in n]))
            prob_l.append(sum([int(ball == l) for ball in n])/k)
        
        # create data for theoretical binomial, poisson and gaussian distributions
        binom = np.random.binomial(N, p, 5000)
        poisson = np.random.poisson(N*p, 5000)
        gaussian = np.random.normal(loc = mean, scale = var, size = 5000)
        
        x_new = np.linspace(0, 21, 100)
        bspline = interpolate.make_interp_spline(range(0,22), prob_l)
        y_new = bspline(x_new)
        
        # sns.histplot(prob_l, label = 'sample')
        sns.lineplot(prob_l, label = 'sample', marker='o', linestyle='')
        sns.lineplot(x=x_new, y=y_new, label ='smoothed sample', linestyle='--', color=[0, 0.4470, 0.7410])
        sns.kdeplot(binom, label = 'binomial', bw_adjust = 2)
        sns.kdeplot(poisson, label = 'poisson', bw_adjust = 2)
        sns.kdeplot(gaussian, label = 'gaussian', bw_adjust = 2)
        plt.legend()
        plt.show()