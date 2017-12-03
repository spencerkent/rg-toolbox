"""
This file defines functionality to radially gaussianize a dataset
"""

import numpy as np

from .whiten import whiten

def apply_rg(raw_data, g_interpolator, whitening_params):
  """
  Radially-gaussianizes a set of datapoints

  Parameters
  ----------
  raw_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset
  g_interpolator : scipy.interpolate.interp1d object
      An object of type scipy.interpolate.interp1d which has been fit according
      to some training data. We can use this to evaluate the function g() for
      this dataset which is used to gaussianize the data
  whitening_params : dict
      Parameters of the whitening transform computed on the training data.
      'PCA_basis' : ndarray
        The (D x D) matrix containing in its columns the eigenvectors of the
        covariance matrix.
      'PCA_axis_variances' : ndarray
        The 1D, size D vector giving variances of the projections onto each PC.

  Returns
  -------
  gaussianized_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset
  radial_scaling : ndarray
      A 1D length N array used to scale the whitened data representation. Will
      be needed if one wants to invert the RG transform
  """
  num_samples = raw_data.shape[1]
  num_components = raw_data.shape[0]

  # first we whiten the data
  white_data = whiten(raw_data, precomputed_params=whitening_params)
  # compute the radius of each point
  radii = np.linalg.norm(white_data, ord=2, axis=0)
  try:
    radial_scaling = g_interpolator(radii) / radii
  except:
    raise ValueError('You should try either extending the support of the ' +
                     'fitted CDF in fit_rg or allow interpolation outside the '+
                     'bounds in the interpolator module')
  # we want to avoid scaling any points exactly to the origin, 0.0, because then
  # we won't be able to invert the transform, so we'll just hackily set the
  # scaling to be very small in that case.
  radial_scaling[radial_scaling == 0.0] = min(np.min(radial_scaling), 1e-8)
  gaussianized_data = radial_scaling[None, :] * white_data
  return gaussianized_data, radial_scaling
