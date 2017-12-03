"""
Implementation of basic PCA whitening
"""
import numpy as np

def whiten(raw_data, precomputed_params=None, return_w_params=False):
  """
  Uses the e-vecs of the covariance matrix to eliminate pairwise correlations

  We can either compute it from scratch or simply apply a precomputed transform

  Parameters
  ----------
  raw_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset
  precomputed_params : dict, optional
      'PCA_basis' : ndarray
        The (D x D) matrix containing in its columns the eigenvectors of the
        covariance matrix.
      'PCA_axis_variances' : ndarray
        The 1D, size D vector giving variances of the projections onto each PC.
  return_w_params : bool, optional
      If True, return the rescaling vector and PCA projection matrix

  Returns
  -------
  whitened_data : ndarray
      Data whitened in the sense of PCA whitening
  whitening_params : dict, (if return_w_params True)
      'PCA_basis' : ndarray
        The (D x D) matrix containing in its columns the eigenvectors of the
        covariance matrix.
      'PCA_axis_variances' : ndarray
        The 1D, size D vector giving variances of the projections onto each PC.
  """
  num_samples = raw_data.shape[1]
  num_components = raw_data.shape[0]
  if precomputed_params is None:
    if num_components > num_samples or num_components > 10**6:
      # If the dimensionality of each datapoint is high, we probably
      # want to compute the SVD of the data directly to avoid forming a huge
      # covariance matrix
      U, s, Vt = np.linalg.svd(raw_data, full_matrices=True)
      whitening_params = {'PCA_basis': U,
                          'PCA_axis_variances': np.square(s) / num_samples}
    else:
      # the SVD is more numerically stable then eig so we'll use it on the
      # covariance matrix directly
      U, w, _ = np.linalg.svd(np.dot(raw_data, raw_data.T) / num_samples,
                              full_matrices=True)
      whitening_params = {'PCA_basis': U, 'PCA_axis_variances': w}
  else:
    # shallow copy just creates a new reference...
    whitening_params = precomputed_params.copy()

  white_data = (np.dot(whitening_params['PCA_basis'].T, raw_data) /
                np.sqrt(whitening_params['PCA_axis_variances'] + 1e-8)[:, None])

  if return_w_params:
    return white_data, whitening_params
  else:
    return white_data


def unwhiten(white_data, w_params):
  """
  Undoes PCA whitening

  Parameters
  ----------
  white_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset
  w_params : dict
      Two ndarrays used to invert the transform
      'PCA_basis' : ndarray
        The (D x D) matrix containing in its columns the eigenvectors of the
        covariance matrix.
      'PCA_axis_variances' : ndarray
        The 1D, size D vector giving variances of the projections onto each PC.
  """
  return np.dot(w_params['PCA_basis'],
                np.sqrt(w_params['PCA_axis_variances'] + 1e-8)[:, None] *
                white_data)
