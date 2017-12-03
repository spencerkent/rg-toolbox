"""
This file defines code to estimate the funtion g() which maps data to a normal
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chi

from matplotlib import pyplot as plt

def estimate_g(radii, data_dimensionality, cdf_extension, cdf_precision):
  """
  Estimates the function g() which is used in a rescaling of the vectors

  Parameters
  ----------
  radii : ndarray
      A 1d array with n elements where n is the number of samples in the dataset
      and each entry is the l2 norm of the representation for that sample
  data_dimensionality : int
      The dimensionality of the datapoints that the radii pertains to. Used to
      set the degrees of freedom on the chi distribution
  cdf_extension : float
      The amount by which to extend the support of the CDF as a fraction of the
      range of radii present in the provided samples
  cdf_precision : int
      The number of samples to devote to evaluating the CDF in the range between
      the maximum and minimum radii

  Returns
  -------
  g_support : ndarray
      A 1d array giving the values of r (radius) at which g has been
      estimated
  g : ndarray
      A 1d array giving the value of g for each value in the support
  """
  g_support, cdf = estimate_radial_CDF(radii, cdf_extension, cdf_precision)
  # the l2 norm of a vector of n iid normal random variables is
  # a random variable distributed as a chi distribution with n degrees of freedom
  # we want it's inverse CDF, or percent point function
  # Because feeding the inverse CDF a value of 1.0 will produce inf, we'll just
  # hackily fix this, slightly lowering the top value of the cdf
  cdf[cdf == 1.0] = cdf[cdf == 1.0] - (1./len(radii))
  g = chi.ppf(cdf, data_dimensionality)
  return g_support, g


def estimate_radial_CDF(radii, support_extension, precision):
  """
  Takes data expressed in terms of euclidean norm and computes the empirical CDF

  This uses the method outlined in Sec 4.2 of L&S 2009. There may be more
  efficient ways to do this.

  Parameters
  ----------
  radii : ndarray
      A 1d array with n elements where n is the number of samples in the dataset
      and each entry is the l2 norm of the representation for that sample
  support_extension : float
      The amount by which to extend the support of the CDF as a fraction of the
      range of radii present in the provided samples
  precision : int
      The number of samples to devote to evaluating the CDF in the range between
      the maximum and minimum radii

  Returns
  -------
  empirical_cdf_support : ndarray
      A 1d array giving the values of r (radius) at which the CDF has been
      estimated
  empirical_cdf : ndarray
      A 1d array giving the estimated CDF for each value in the support
  """
  n_samps = len(radii)
  sorted_indices = np.argsort(radii)  # quicksort
  sorted_radii = radii[sorted_indices]
  min_radius = sorted_radii[0]
  max_radius = sorted_radii[-1]
  extension_amount = support_extension * abs(max_radius - min_radius)
  num_extension_bins = int(precision * support_extension)

  # this counts the number of radii that were less than or equal to a specific
  # radius over the full range of radii sampled at the specified precision
  counts = np.searchsorted(sorted_radii,
                           np.linspace(min_radius, max_radius, precision),
                           side='right')
  counts = counts / n_samps

  empirical_cdf_support = np.linspace(min_radius - extension_amount,
                                      max_radius + extension_amount,
                                      num_extension_bins * 2 + precision)
  if num_extension_bins != 0:
    empirical_cdf = np.zeros(len(empirical_cdf_support))
    empirical_cdf[num_extension_bins:-num_extension_bins] = counts
    empirical_cdf[-num_extension_bins:] = 1.
  else:
    empirical_cdf = np.copy(counts)

  return empirical_cdf_support, empirical_cdf
