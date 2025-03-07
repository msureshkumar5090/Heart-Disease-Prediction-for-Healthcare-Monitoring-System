import numpy as np
import time


def GSOA(S, fitness_function, lb, ub, max_iter):
    """
    Gater Snake Optimization Algorithm (GSO)
    """
    N, dim = S.shape
    best_solution = np.zeros(dim)
    best_fit = float("inf")
    # Define male, female, and she-male count
    N_m = N // 3  # Males
    N_f = N // 3  # Females
    N_sm = N - (N_m + N_f)  # She-males
    # Alpha and Beta parameters for movement
    alpha = 0.5
    beta = 0.1

    fitness_history = []
    ct = time.time()
    for t in range(max_iter):
        # Calculate fitness
        fitness = np.array([fitness_function(s) for s in S])

        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = S[best_idx]
        best_fit = fitness[best_idx]
        fitness_history.append(best_fit)

        # Step 3: Calculate weights
        worst_fitness = np.max(fitness)
        best_fitness = np.min(fitness)
        weights = (worst_fitness - fitness) / (worst_fitness - best_fitness + 1e-6)

        # Step 4: Move Males
        for i in range(N_m):
            I_j = np.random.uniform(-1, 1, dim)  # Random influence
            S[N_f + i] += alpha * I_j * (weights[N_f + i] / np.sum(weights)) + beta

        # Step 5: Move Females
        for i in range(N_f):
            I_j = np.random.uniform(-1, 1, dim)
            S[i] += alpha * I_j * (best_solution - S[i]) + beta

        # Step 6: Move She-males
        for i in range(N_sm):
            S[N_f + N_m + i] += beta
        S = np.clip(S, lb, ub)

        if np.abs(best_fitness - worst_fitness) < 1e-5:
            break
    ct = time.time() - ct
    return best_fit, fitness_history, best_solution, ct

