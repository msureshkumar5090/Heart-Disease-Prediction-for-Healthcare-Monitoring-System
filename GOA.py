import numpy as np
import random
import time


# Levy Flight Function
def levy(n, m, beta):
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(n, m) * sigma
    v = np.random.randn(n, m)
    step = u / abs(v) ** (1 / beta)
    return step


# Gazelle Optimization Algorithm Function
def GOA(gazelle, fobj, Xmin, Xmax, Max_iter):
    SearchAgents_no, dim = gazelle.shape
    Top_gazelle_pos = np.zeros(dim)
    Top_gazelle_fit = float("inf")
    Convergence_curve = np.zeros(Max_iter)
    stepsize = np.zeros((SearchAgents_no, dim))
    fitness = np.full((SearchAgents_no, 1), float("inf"))
    PSRs = 0.34
    S = 0.88
    s = np.random.rand()
    Iter = 0
    ct = time.time()
    while Iter < Max_iter:
        # Evaluate fitness of each gazelle
        for i in range(SearchAgents_no):
            gazelle[i, :] = np.clip(gazelle[i, :], Xmin[i, :], Xmax[i, :])
            fitness[i, 0] = fobj(gazelle[i, :])
            if fitness[i, 0] < Top_gazelle_fit:
                Top_gazelle_fit = fitness[i, 0]
                Top_gazelle_pos = gazelle[i, :].copy()
        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = gazelle.copy()
        # Update Prey positions
        Inx = fit_old < fitness
        gazelle = np.where(Inx, Prey_old, gazelle)
        fitness = np.where(Inx, fit_old, fitness)
        fit_old = fitness.copy()
        Prey_old = gazelle.copy()
        Elite = np.tile(Top_gazelle_pos, (SearchAgents_no, 1))
        CF = (1 - Iter / Max_iter) ** (2 * Iter / Max_iter)
        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)
        RB = np.random.randn(SearchAgents_no, dim)
        for i in range(SearchAgents_no):
            for j in range(dim):
                R = np.random.rand()
                r = np.random.rand()
                mu = -1 if Iter % 2 == 0 else 1
                if r > 0.5:
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * gazelle[i, j])
                    gazelle[i, j] += s * R * stepsize[i, j]
                else:
                    if i > SearchAgents_no / 2:
                        stepsize[i, j] = RB[i, j] * (RL[i, j] * Elite[i, j] - gazelle[i, j])
                        gazelle[i, j] = Elite[i, j] + S * mu * CF * stepsize[i, j]
                    else:
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * gazelle[i, j])
                        gazelle[i, j] += S * mu * R * stepsize[i, j]
        # Evaluate new fitness
        for i in range(SearchAgents_no):
            gazelle[i, :] = np.clip(gazelle[i, :],  Xmin[i, :], Xmax[i, :])
            fitness[i, 0] = fobj(gazelle[i, :])
            if fitness[i, 0] < Top_gazelle_fit:
                Top_gazelle_fit = fitness[i, 0]
                Top_gazelle_pos = gazelle[i, :].copy()
        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = gazelle.copy()
        Inx = fit_old < fitness
        gazelle = np.where(Inx, Prey_old, gazelle)
        fitness = np.where(Inx, fit_old, fitness)
        fit_old = fitness.copy()
        Prey_old = gazelle.copy()

        if random.random() < PSRs:
            U = np.random.rand(SearchAgents_no, dim) < PSRs
            gazelle = gazelle + CF * ((Xmin + np.random.rand(SearchAgents_no, dim) * (Xmax - Xmin)) * U)
        else:
            r = random.random()
            Rs = SearchAgents_no
            stepsize = (PSRs * (1 - r) + r) * (
                        gazelle[np.random.permutation(Rs), :] - gazelle[np.random.permutation(Rs), :])
            gazelle = gazelle + stepsize
        Convergence_curve[Iter] = Top_gazelle_fit
        Iter += 1
    ct = time.time() - ct
    return Top_gazelle_fit, Convergence_curve, Top_gazelle_pos, ct
