import numpy as np

def double_logistic(x: np.ndarray[float], kup: float, kdn: float, betaup: float, betadn: float):
    '''
    Express annual phenological parameters as a logistic curve with rising and descending arms.

    kup: Slope of rising arm.
    kdn: Slope of falling arm.
    betaup: Inflection point of rising arm.
    betadn: Inflection point of falling arm.
    '''
    log1 = 1 / (1 + np.exp(-kup * (x - betaup)))
    log2 = 1 / (1 + np.exp(-kdn * (x - betadn)))
    return(log1 * (1-log2))

def multiyear_double_logistic(kup: np.ndarray[float], kdn: np.ndarray[float], betaup: np.ndarray[float], betadn: np.ndarray[float]) -> np.ndarray[float]:
    '''
    Combine phenological parameters from multiple years in one timeseries.
    '''
    assert all(kup.shape == a.shape for a in (kup, kdn, betaup, betadn))

    n_years = kup.shape[0]

    x = np.linspace(-100, 365*n_years, num=100*n_years)
    y_out = np.zeros_like(x)

    for i in range(n_years):
        x_shift = x - (i * 365)
        y_out += double_logistic(x_shift, kup[i], kdn[i], betaup[i], betadn[i])

    return y_out

if __name__ == "__main__":
    kup = np.array([0.1, 0.1, 0.1, 0.1])
    kdn = np.array([0.1, 0.1, 0.1, 0.1])
    betaup = np.array([100, 100, 100, 100])
    betadn = np.array([100, 100, 100, 100])

    y_out = multiyear_double_logistic(kup, kdn, betaup, betadn)
