"""
This file defines functionality to INVERT the radial gaussianization of data
"""
import numpy as np

from .whiten import unwhiten

def invert_rg(gaussianized_data, orig_scaling, orig_whitening_params):
  """
  Applies the inverse of the RG transform to gaussian data

  Parameters
  ----------
  gaussianized_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset
  orig_scaling : ndarray
      A N-dimensional vector giving the radial scaling of the data under the 
      forward RG transform
  orig_whitening_params : dict
      Used to project the data back into the input space
      'PCA_basis' : ndarray
        The (D x D) matrix containing in its columns the eigenvectors of the
        covariance matrix.
      'PCA_axis_variances' : ndarray
        The 1D, size D vector giving variances of the projections onto each PC.
  """
  white_data = gaussianized_data / orig_scaling[None, :]
  # Now we invert the whitening transform
  return unwhiten(white_data, orig_whitening_params)
