"""
Functionality to fit the parameters of a radial gaussianization transform
"""

import numpy as np
from scipy.interpolate import interp1d

from .estimate_g import estimate_g
from .whiten import whiten

def fit_rg(raw_data):
  """
  Fit the full RG transform which includes the PCA whitening transform

  Parameters
  ----------
  raw_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset

  Returns
  -------
  g_interpolator : scipy.interpolate.interp1d object
      An object of type scipy.interpolate.interp1d which has been fit according
      to this training data. We can use this to evaluate the function g() on
      new arbitrary points
  whitening_params : dict
      Parameters of the whitening transform computed on the training data.
      Also needed if one wishes to fully invert the RG transform
      'PCA_basis' : ndarray
        The (D x D) matrix containing in its columns the eigenvectors of the
        covariance matrix.
      'PCA_axis_variances' : ndarray
        The 1D, size D vector giving variances of the projections onto each PC.
  """
  num_samples = raw_data.shape[1]
  num_components = raw_data.shape[0]

  # first we whiten the data
  white_data, whitening_params = whiten(raw_data, return_w_params=True)
  # compute the radius of each point
  radii = np.linalg.norm(white_data, ord=2, axis=0)
  # estimate the scaling function g()
  g_est_support, g_est = estimate_g(radii, num_components,
                                    1.0, num_samples // 10)
  # to evaluate the function g at each of the given radii, we use interpolation
  return interp1d(g_est_support, g_est), whitening_params
