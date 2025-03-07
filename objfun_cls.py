import numpy as np
from Global_Vars import Global_Vars
from RELIEF import reliefF


def objfun(Soln):
    Feat_1 = Global_Vars.Feat_1
    Feat_2 = Global_Vars.Feat_2
    Feat_3 = Global_Vars.Feat_3
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Weighted_feat_1 = Feat_1 * sol[:Feat_1.shape[-1]]
            Weighted_feat_2 = Feat_2 * sol[Feat_1.shape[-1]:Feat_1.shape[-1] + Feat_2.shape[-1]]
            Weighted_feat_3 = Feat_3 * sol[Feat_1.shape[-1] + Feat_2.shape[-1]:]
            Weighted_Fused_Feat = np.concatenate((Weighted_feat_1, Weighted_feat_2, Weighted_feat_3), axis=1)
            score = reliefF(np.asarray(Weighted_Fused_Feat), np.ravel(Tar))
            Fitn[i] = 1 / score
        return Fitn
    else:
        sol = Soln
        Weighted_feat_1 = Feat_1 * sol[:Feat_1.shape[-1]]
        Weighted_feat_2 = Feat_2 * sol[Feat_1.shape[-1]:Feat_1.shape[-1] + Feat_2.shape[-1]]
        Weighted_feat_3 = Feat_3 * sol[Feat_1.shape[-1] + Feat_2.shape[-1]:]
        Weighted_Fused_Feat = np.concatenate((Weighted_feat_1, Weighted_feat_2, Weighted_feat_3), axis=1)
        score = reliefF(np.asarray(Weighted_Fused_Feat), np.ravel(Tar))
        Fitn = 1 / score
        return Fitn
