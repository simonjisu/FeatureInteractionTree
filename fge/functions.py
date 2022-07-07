import numpy as np

def g_base(siv: np.ndarray, build_global: bool):
    """base case
    scores = shap interaction values 
    
    `build_global` will work differently compare to `g_abs`, 
    because it applies the absoulute values after calculate mean values

    Args:
        siv (np.ndarray): shap interaction values, size of NxFxF, 
            where F is number of features, N is number of instances 
        build_global (bool): where building global interaction tree 
    """    
    if build_global:
        siv_scores = siv.mean(0)
    else:
        siv_scores = siv
    return np.abs(siv_scores)

def g_abs(siv: np.ndarray, build_global: bool):
    """abs case
    scores = absolute shap interaction values

    `build_global` will work differently compare to `g_base`, 
    because it applies the absoulute values before calculate mean values

    Args:
        siv (np.ndarray): shap interaction values, size of NxFxF, 
            where F is number of features, N is number of instances 
        build_global (bool): where building global interaction tree 
    """    
    if build_global:
        siv_scores = np.abs(siv).mean(0)
    else:
        siv_scores = np.abs(siv)
    return siv_scores

def g_abs_only_interaction(siv: np.ndarray, build_global: bool):
    """abs case
    scores = absolute shap interaction values only considering interaction terms
    only considering the interaction terms.

    example: let's say we have shap interaction values of two features(A, B)
         A    B
    A   0.5  -0.1
    B  -0.1   0.3

    -> 1. Calculate absolute scores
         A    B
    A   0.5   0.1
    B   0.1   0.3
    -> 2. Fill all diag indices with 0.
         A    B
    A   0.0  0.1
    B   0.1  0.0

    Args:
        siv (np.ndarray): shap interaction values, size of NxFxF, 
            where F is number of features, N is number of instances 
        build_global (bool): where building global interaction tree 
    """
    n_features = siv.shape[-1]

    if build_global:
        siv_scores = np.abs(siv).mean(0)
    else:
        siv_scores = np.abs(siv)
    r_diag, c_diag = np.diag_indices(n_features)
    siv_scores[r_diag, c_diag] = 0.0
    
    return siv_scores

def g_ratio(siv: np.ndarray, build_global: bool):
    """relative ratio case
    scores = ratio of absolute shap interaction values to main effects
    only considering the interaction term between.

    example: let's say we have shap interaction values of two features(A, B)
         A    B
    A   0.5  -0.1
    B  -0.1   0.3

    -> 1. Calculate absolute scores
         A    B
    A   0.5   0.1
    B   0.1   0.3
    -> 2. Interaction Relative Effect Ratio of (A, B) = 0.1 / (0.5 + 0.3) = 1 / 8 = 0.125
    -> 3. Fill all diag indices with 0.
         A    B
    A   0.0   0.125
    B   0.125 0.0

    Args:
        siv (np.ndarray): shap interaction values, size of NxFxF, 
            where F is number of features, N is number of instances 
        build_global (bool): where building global interaction tree 
    """
    n_features = siv.shape[-1]
    if build_global:
        siv_scores = np.abs(siv).mean(0)
    else:
        siv_scores = np.abs(siv)

    r_l, c_l = np.tril_indices(n_features, -1)
    for r, c in list(zip(r_l, c_l)):
        cmb_main_effect = siv_scores[r, r] + siv_scores[c, c]
        # check zeros
        if cmb_main_effect < 1e-8:
            cmb_main_effect += 1e-8
        siv_scores[r, c] = siv_scores[r, c] / cmb_main_effect
        
    r_u, c_u = zip(*(zip(*(c_l, r_l))))
    siv_scores[r_u, c_u] = siv_scores[r_l, c_l]
    r_diag, c_diag = np.diag_indices(n_features)
    siv_scores[r_diag, c_diag] = 0.0
    
    return siv_scores