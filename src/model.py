import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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

