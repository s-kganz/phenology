import numpy as np

def double_logistic(x: np.ndarray[float], kup: float, kdn: float, betaup: float, betadn: float):
    log1 = 1 / (1 + np.exp(-kup * (x - betaup)))
    log2 = 1 / (1 + np.exp(-kdn * (x - betadn)))
    return(log1 * (1-log2))