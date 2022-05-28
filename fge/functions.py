import numpy as np

def g_base(siv: np.ndarray, build_global: bool):
    """base case
    scores = absolute shap interaction values 
    
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

def g_ratio(siv: np.ndarray, build_global: bool):
    """ratio case
    scores = ratio of absolute shap interaction values to main effects

    Args:
        siv (np.ndarray): shap interaction values, size of NxFxF, 
            where F is number of features, N is number of instances 
        build_global (bool): where building global interaction tree 
    """
    n_features = siv.shape[-1]
    r, c = np.diag_indices(n_features)
    if build_global:
        siv_scores = np.abs(siv).mean(0)
    else:
        siv_scores = np.abs(siv)
    siv_scores_diag = siv_scores[r, c]
    siv_scores = siv_scores / siv_scores_diag
    return siv_scores