import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CircularTransformer(BaseEstimator, TransformerMixin):
    '''
    Implements an angular transformation of the target variable. This transformation
    is used in, for example:

    https://bg.copernicus.org/articles/17/3991/2020/
    '''
    def __init__(self, cycle_length=365):
        self.cycle_length = cycle_length
        self.to_rad_ = (2 * np.pi) / cycle_length
        self.mean_angle_ = None

    def to_radians(self, x):
        return x * self.to_rad_

    def from_radians(self, x):
        return x / self.to_rad_
        
    def fit(self, X, y=None):
        self.mean_angle_ = np.mean(self.to_radians(X))
        return self
        
    def transform(self, X, y=None):
        return np.tan((self.to_radians(X) - self.mean_angle_) / 2)

    def inverse_transform(self, X, y=None):
        return self.from_radians(2 * np.atan(X) + self.mean_angle_)

def jammalamadaka_sharma_corr(alpha, beta, cycle_length=365):
    """
    Computes the Jammalamadaka-Sharma circular correlation coefficient.
    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)

    alpha = alpha * (2 * np.pi) / cycle_length
    beta  = beta * (2 * np.pi) / cycle_length
    
    # Compute circular means using mean of sines and cosines
    mean_alpha = np.arctan2(np.mean(np.sin(alpha)), np.mean(np.cos(alpha)))
    mean_beta = np.arctan2(np.mean(np.sin(beta)), np.mean(np.cos(beta)))
    
    # Compute deviations
    sin_alpha_dev = np.sin(alpha - mean_alpha)
    sin_beta_dev = np.sin(beta - mean_beta)
    
    # Calculate numerator and denominator components
    num = np.sum(sin_alpha_dev * sin_beta_dev)
    den = np.sqrt(np.sum(sin_alpha_dev**2) * np.sum(sin_beta_dev**2))
    
    if den == 0:
        return 0.0
    return num / den

def angular_absolute_error(y_true, y_hat, cycle_length=365):
    to_rad = (2 * np.pi) / cycle_length
    
    y_true_rad = y_true * to_rad
    y_hat_rad  = y_hat  * to_rad

    error_rad = np.minimum(np.abs(y_true_rad - y_hat_rad), 2*np.pi - np.abs(y_true_rad - y_hat_rad))

    return error_rad / to_rad

def angular_root_mean_square_error(y_true, y_hat, cycle_length=365):
    residual = angular_absolute_error(y_true, y_hat, cycle_length)
    return np.sqrt(np.mean(residual ** 2))

def model_performance(y_true, y_hat, cycle_length=365):
    return pd.DataFrame(data=dict(
        metric=["js_correlation", "angular_mae", "angular_rmse"],
        value=[
            jammalamadaka_sharma_corr(y_true, y_hat, cycle_length),
            angular_absolute_error(y_true, y_hat, cycle_length).mean(),
            angular_root_mean_square_error(y_true, y_hat, cycle_length)
        ]
    ))

def greenup_activity(
    temperature: np.ndarray[float], precipitation: np.ndarray[float], 
    tmin: float, 
    temp_sigmoid_inflection: float, temp_sigmoid_slope: float,
    prcp_sigmoid_inflection: float, prcp_sigmoid_slope: float,
    verbose: bool=False
):
    """Predict vegetation activity from cumulative temperature and precipitation forcing.

    Temperature forcing accumulates as degree-days above `tmin`. All precipitation
    contributes directly to forcing.

    Parameters
    ----------
    temperature : array-like
        Daily air temperature time series.
    precipitation : array-like
        Daily precipitation time series, same length as `temperature`.
    tmin : array-like
        Minimum temperature threshold for degree-day accumulation (fitted parameter).
    (temp|prcp)_sigmoid_(inflection|slope) : array-like
        Fitted parameters controlling the cumulative forcing function.
    verbose : bool
        Whether to return the temp/prcp forcing function output alongside overall
        vegetation activity.

    Returns
    -------
    array-like
        Time series of predicted vegetation activity in the range 0-1. If `verbose`,
        returns `(temp_forcing, prcp_forcing, total_forcing)`.
    """
    #temperature = np.asarray(temperature, dtype=float)
    #precipitation = np.asarray(precipitation, dtype=float)

    if temperature.shape != precipitation.shape:
        raise ValueError("temperature and precipitation must have the same shape")

    # Degree-day temperature forcing, clipped at zero
    temp_forcing = np.maximum(temperature - tmin, 0.0)
    temp_forcing_cumulative = np.cumsum(temp_forcing, axis=1)

    # Precipitation forcing is just the sum
    prcp_forcing_cumulative = np.cumsum(precipitation, axis=1)

    # Calculate sigomid responses
    temp_forcing_sigmoid = expit(np.exp(temp_sigmoid_slope) * (temp_forcing_cumulative - temp_sigmoid_inflection))
    prcp_forcing_sigmoid = expit(np.exp(prcp_sigmoid_slope) * (prcp_forcing_cumulative - prcp_sigmoid_inflection))

    # Multiply
    total_forcing = temp_forcing_sigmoid * prcp_forcing_sigmoid

    if verbose:
        return temp_forcing_sigmoid, prcp_forcing_sigmoid, total_forcing
    else:
        return total_forcing

def sos_swq_tmin(
    temperature: np.ndarray[float], precipitation: np.ndarray[float],
    b: float, c: float, # precip sigmoid
    Preq: float, # cumulative precip requirement
    Tthres: float, Tb: float, # temperature forcing parameters
    Fcrit: float # total forcing parameter
):
    # Check whether precipitation forcing meets Preq
    P_sigmoid = 1 / (1 + np.exp(-b * (precipitation - c)))
    Sw = np.cumsum(P_sigmoid, axis=-1)
    k = np.where(Sw > Preq, 1, 0)

    # Accumulate temperature forcing
    T_forcing = k * np.maximum(0, temperature - Tb)
    Sf = np.cumsum(T_forcing, axis=-1)

    # Reset the cumsum when temprature < Tthres
    reset_vals = np.zeros_like(T_forcing)
    reset_indices = np.where(temperature < Tthres)
    reset_vals[reset_indices] = Sf[reset_indices]
    subtract_mask = np.maximum.accumulate(reset_vals, axis=-1)
    Sf = Sf - subtract_mask
    
    # Return running arrays so we can verify that this works
    return (
        P_sigmoid,
        Sw,
        k,
        T_forcing,
        Sf
    )


def sos_swq_tmin_predictor(
    temperature: np.ndarray[float], precipitation: np.ndarray[float],
    b: float, c: float, # precip sigmoid
    Preq: float, # cumulative precip requirement
    Tthres: float, Tb: float, # temperature forcing parameters
    Fcrit: float # total forcing parameter
):
    _, _, _, _, Sf = sos_swq_tmin(temperature, precipitation, b, c, Preq, Tthres, Tb, Fcrit)

    return np.argmax(Sf > Fcrit, axis=-1)


def sos_swq_tmin_duration(
    temperature: np.ndarray[float], precipitation: np.ndarray[float],
    b: float, c: float, # precip sigmoid
    Preq: float, # cumulative precip requirement
    Tthres: float, Tb: float, # temperature forcing parameters
    Fcrit: float # total forcing parameter
):
    '''
    Get the time to reach Preq and Fcrit in the sos_swq_tmin model
    '''
    P_sigmoid, Sw, k, T_forcing, Sf = sos_swq_tmin(temperature, precipitation, b, c, Preq, Tthres, Tb, Fcrit)

    time_to_preq  = np.argmax(k, axis=-1) + 1
    time_to_fcrit = np.argmax(Sf > Fcrit, axis=-1) + 1 - time_to_preq

    return time_to_preq, time_to_fcrit
