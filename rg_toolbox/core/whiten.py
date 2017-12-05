"""
Implementation of basic PCA whitening
"""
import numpy as np

def whiten(raw_data, w_type='PCA', precomputed_params=None,
           return_w_params=False):
  """
  Uses the e-vecs of the covariance matrix to eliminate pairwise correlations

  We can either compute it from scratch or simply apply a precomputed transform

  Parameters
  ----------
  raw_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset
  w_type : str, optional
      The type of whitening. For now, you can choose between 'PCA' and 'ZCA'.
      This input is ignored if precomputed_params is provided, we'll look for
      the parameter there.
  precomputed_params : dict, optional
      'w_type' : The type of transform to use. Overrides w_type input.
        Either 'PCA' or 'ZCA'
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
      'w_type' : The type of transform used. Either 'PCA' or 'ZCA'
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
                          'PCA_axis_variances': np.square(s) / num_samples,
                          'w_type': w_type}
    else:
      # the SVD is more numerically stable then eig so we'll use it on the
      # covariance matrix directly
      U, w, _ = np.linalg.svd(np.dot(raw_data, raw_data.T) / num_samples,
                              full_matrices=True)
      whitening_params = {'PCA_basis': U, 'PCA_axis_variances': w,
                          'w_type': w_type}
  else:
    # shallow copy just creates a new reference...
    whitening_params = precomputed_params.copy()

  if whitening_params['w_type'] == 'PCA':
    white_data = \
        (np.dot(whitening_params['PCA_basis'].T, raw_data) /
         np.sqrt(whitening_params['PCA_axis_variances'] + 1e-8)[:, None])
  elif whitening_params['w_type'] == 'ZCA':
    # this type of whitening produces the samples that are closest to the raw
    # subject to the whitening constraint on the correlation matrix
    white_data = \
        np.dot(whitening_params['PCA_basis'],
               (np.dot(whitening_params['PCA_basis'].T, raw_data) /
                np.sqrt(whitening_params['PCA_axis_variances'] + 1e-8)[:, None]))
  else:
    raise KeyError('Unrecognized whitening type ' + w_type)

  if return_w_params:
    return white_data, whitening_params
  else:
    return white_data


def unwhiten(white_data, w_params):
  """
  Undoes whitening

  Parameters
  ----------
  white_data : ndarray
      A (D x N) array where D is the dimensionality of each datapoint and N is
      the number of datapoints in our dataset
  w_params : dict
      Two ndarrays used to invert the transform
      'w_type' : The type of transform used. Either 'PCA' or 'ZCA'
      'PCA_basis' : ndarray
        The (D x D) matrix containing in its columns the eigenvectors of the
        covariance matrix.
      'PCA_axis_variances' : ndarray
        The 1D, size D vector giving variances of the projections onto each PC.
  w_type : str
      The type of whitening. For now, you can choose between 'PCA' and 'ZCA'
  """
  if w_params['w_type'] == 'PCA':
    return np.dot(w_params['PCA_basis'],
                  np.sqrt(w_params['PCA_axis_variances'] + 1e-8)[:, None] *
                  white_data)
  elif w_params['w_type'] == 'ZCA':
    return np.dot(w_params['PCA_basis'],
                  np.sqrt(w_params['PCA_axis_variances'] + 1e-8)[:, None] *
                  np.dot(w_params['PCA_basis'].T, white_data))
  else:
    raise KeyError('Unrecognized whitening type ' + w_params['w_type'])

