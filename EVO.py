import time
import numpy as np


def distance(a, b):
    o = np.zeros((a.shape[1 - 1]))
    for i in range(a.shape[1 - 1]):
        o[i] = np.sqrt((a[i] - b[i]) ** 2)
    return o


def EVO(Particles, CostFunction, VarMin, VarMax, MaxFes):
    nParticles, VarNumber = Particles.shape[0], Particles.shape[1]
    NELs = np.zeros((nParticles))
    for i in range(nParticles):
        NELs[i] = CostFunction(Particles[i, :])
    NELs, SortOrder = np.sort(NELs), np.argsort(NELs)
    Particles = Particles[SortOrder, :]
    BS = Particles[0, :]
    BS_NEL = NELs[0]
    WS_NEL = NELs[-1]
    BestCosts = np.ones((MaxFes))
    ct = time.time()
    for FEs in range(MaxFes):
        NewParticles = []
        NewNELs = []
        Dist = np.zeros((nParticles, VarNumber))
        for i in range(nParticles):
            for j in range(VarNumber):
                Dist[i] = distance(Particles[i], Particles[i])
                __, a = np.sort(Dist), np.argsort(Dist)

                CnPtIndex = np.random.randint(nParticles)
                if CnPtIndex < 3:
                    CnPtIndex = CnPtIndex + 2
                CnPtA = Particles[a[np.arange(CnPtIndex + 1)], :]
                CnPtB = NELs[a[np.arange(CnPtIndex), :]]
                X_NG = np.mean(CnPtA)
                X_CP = np.mean(Particles)
                EB = np.mean(NELs)
                SL = (NELs[i] - BS_NEL) / (WS_NEL - BS_NEL)
                SB = np.random.rand()
                NewNEL = np.zeros((nParticles))
                NewParticle = np.zeros((nParticles, VarNumber))
                if NELs[i] > EB:
                    if SB > SL:
                        AlphaIndex1 = np.random.randint(VarNumber)
                        AlphaIndex2 = np.random.randint(AlphaIndex1, np.array([VarNumber]), 1)
                        NewParticle[i, :] = Particles[i, :]
                        NewParticle[i, AlphaIndex2] = BS[AlphaIndex2]
                        GamaIndex1 = np.random.randint(VarNumber)
                        GamaIndex2 = np.random.randint(GamaIndex1, np.array([VarNumber]), 1)
                        NewParticle[2, :] = Particles[i, :]
                        NewParticle[2, GamaIndex2[0]] = X_NG
                        NewNEL[1] = CostFunction(NewParticle[1, :])
                        NewNEL[2] = CostFunction(NewParticle[2, :])
                    else:
                        Ir = np.random.uniform(0, 1, 1)
                        Jr = np.random.uniform(0, 1, 1)
                        NewParticle[i, :] = Particles[i, :] + (np.multiply(Jr[0], (Ir[0] * BS[j] - Ir[0] * X_CP)) / SL)
                        Ir = np.random.uniform(0, 1, 1)
                        Jr = np.random.uniform(0, 1, 1)
                        NewParticle[i, :] = Particles[i, j] + (np.multiply(Jr[0], (Ir[0] * BS[j] - Ir[0] * X_NG)))

                        NewNEL[i] = CostFunction(NewParticle[i])

                else:
                    NewParticle[i, :] = Particles[i, :] + np.random.rand() * SL * np.random.uniform(VarMin[i],
                                                                                                    VarMax[i], np.array(
                            [1, VarNumber]))

                    NewNEL[i] = CostFunction(NewParticle[i])

                NewParticles = NewParticle
                NewNELs = [NewNEL]
        NewParticles = NewParticles
        NewNELs = NewNELs
        # Sort Particles
        NewNELs, SortOrder = np.sort(NewNELs), np.argsort(NewNELs)
        NewParticles = NewParticles[SortOrder, :]
        Particles = NewParticles[0, np.arange(nParticles), :]
        NELs = NewNELs[0, np.arange(nParticles)]
        # Store Best Cost Ever Found
        BestCosts[FEs] = np.min(BS_NEL)
    Conv_History = BestCosts
    Best_Pos = BS
    Best_fit = np.min(BS_NEL)
    ct = time.time() - ct
    return Best_fit, Conv_History, Best_Pos, ct

