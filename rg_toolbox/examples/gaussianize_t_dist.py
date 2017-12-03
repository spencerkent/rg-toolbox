"""
Demonstrates the use of Radial Gaussianization on a Student's T distribution
"""
import sys
rg_install_dir = '/Users/spencer.kent/Software_staging_area/rg-toolbox/'
sys.path.append(rg_install_dir)

import numpy as np
import itertools
from statsmodels.sandbox.distributions import multivariate
#^ scipy stats doesn't have a multivariate t distribution, although we could
# also create one ourselves
from scipy.stats import kurtosis
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns

from rg_toolbox.core.fit_rg import fit_rg
from rg_toolbox.core.apply_rg import apply_rg
from rg_toolbox.core.invert_rg import invert_rg

def run_rg():
  # parameters for sampled data
  num_samples_train = 10000
  num_samples_test = 10000
  num_components = 10
  # This "sigma" matrix parameterizes the elliptically-symmetric
  # distribution. For a gaussian ESD it is equivalent to the covariance matrix
  temp = np.random.multivariate_normal(np.ones(num_components),
                                       np.eye(num_components),
                                       size=num_components)
  sigma = np.dot(temp, temp.T) + np.diag(1e-5*np.ones(num_components))
  #^ create random positive definite matrix

  ########################################
  # Create a dataset to train the model on
  ########################################
  # *** our convention is that first dimension indexes components, and ***
  # *** second dimension indexes samples ***
  # Draw samples from a general multivariate student's t distribution
  t_samps = multivariate.multivariate_t_rvs(
      np.zeros(num_components), sigma, num_components/2, num_samples_train).T
  # center the data
  t_samps = t_samps - np.mean(t_samps, axis=1)[:, None]

  # Draw samples from a multivariate normal to compare against
  g_samps = np.random.multivariate_normal(
      np.zeros(num_components), sigma, size=num_samples_train).T
  # center the data
  g_samps = g_samps - np.mean(g_samps, axis=1)[:, None]


  ######################################
  # fit RG model to the training dataset
  ######################################
  g_func_fit, p_whitening = fit_rg(t_samps)

  # Apply the model to the training set as a sanity check
  rg_t_samps, scalings = apply_rg(t_samps, g_func_fit, p_whitening)

  # Invert the transformation to get back our original samples.
  reconstructed_t_samps = invert_rg(rg_t_samps, scalings, p_whitening)


  ##############################################
  # Create some testing data and apply the model
  ##############################################
  # Draw samples from the same general multivariate student's t distribution
  t_samps_test = multivariate.multivariate_t_rvs(
      np.zeros(num_components), sigma, num_components/2, num_samples_test).T
  # center the data
  t_samps_test = t_samps_test - np.mean(t_samps_test, axis=1)[:, None]

  rg_t_samps_test, scalings_test = apply_rg(t_samps_test, g_func_fit,
                                            p_whitening)
  reconstructed_t_samps_test = invert_rg(rg_t_samps_test, scalings_test,
                                         p_whitening)
  # we can actually synthesize new data too
  synth_rg_samps = np.random.multivariate_normal(
      np.zeros(num_components), np.eye(num_components), size=num_samples_test).T
  # We'll choose the scalings as samples from an estimate of the
  # appropriate scalings for the distribution we want to map to,
  # in this case the general multivariate student's T distribution.
  synth_use_KDE = False
  #^ We can either compute a kernel density estimate of the distribution of
  #  scalings, or we can just resample the scalings in our training set.
  #  Kernel density estimation can be slow and innaccurate, so for speed choose
  #  the resampling
  if synth_use_KDE:
    #### We could try to determine the best bandwidth
    # params = {'bandwidth': np.logspace(-1, 1, 20)}
    # grid = GridSearchCV(KernelDensity(), params)
    # grid.fit(scalings.reshape(-1, 1))
    # print("KDE estimate for scaling - best bandwidth: ",
    #       grid.best_estimator_.bandwidth)
    # kde = grid.best_estimator_
    #### or just plug something in
    kde = KernelDensity(bandwidth=0.127)
    kde.fit(scalings)
    sampled_scalings = np.squeeze(kde.sample(num_samples_test))
  else:
    sampled_scalings = np.random.choice(scalings, num_samples_test,
                                        replace=False)

  synth_samps = invert_rg(synth_rg_samps, sampled_scalings, p_whitening)


  #######################
  # Plotting some results
  #######################
  print("Plotting...")
  # we will use a few randomly-chosen 2D projections of our multivariate data
  # to visualize the distributions of our data
  if num_components < 2:
    raise ValueError('Univariate!')
  if num_components == 2:
    plotted_component_pairs = [[0, 1]]
  else:
    # pick 3 random pairs of components
    plotted_component_pairs = []
    inds = np.arange(num_components)
    for _ in range(3):
      first_c = np.random.choice(inds)
      second_c = np.random.choice(np.delete(inds, first_c))
      while ([first_c, second_c] in plotted_component_pairs or
             [second_c, first_c] in plotted_component_pairs):
        second_c = np.random.choice(np.delete(inds, first_c))
      plotted_component_pairs.append([first_c, second_c])

  # gather variance and kurtosis stats for these pairs
  stats_t_train = {} # stats for the training set
  stats_t_test = {} # stats for the testing set
  stats_g = {} # stats for the gaussian reference set
  stats_synth = {} # stats for the synthetic data set
  for pair_idx in range(3):
    c1 = plotted_component_pairs[pair_idx][0]
    c2 = plotted_component_pairs[pair_idx][1]
    stats_t_train[pair_idx] = {
        'input': compute_var_kurt(t_samps[[c1, c2]]),
        'rg': compute_var_kurt(rg_t_samps[[c1, c2]]),
        'rec_input': compute_var_kurt(reconstructed_t_samps[[c1, c2]])}
    stats_t_test[pair_idx] = {
        'input': compute_var_kurt(t_samps_test[[c1, c2]]),
        'rg': compute_var_kurt(rg_t_samps_test[[c1, c2]]),
        'rec_input': compute_var_kurt(reconstructed_t_samps_test[[c1, c2]])}
    stats_g[pair_idx] = compute_var_kurt(g_samps[[c1, c2]])
    stats_synth[pair_idx] = {
        'rg': compute_var_kurt(synth_rg_samps[[c1, c2]]),
        'rec_input': compute_var_kurt(synth_samps[[c1, c2]])}

  # compare our training data to data from a gaussian distribution
  plt.figure(figsize=(10, 15))
  for pair_idx in range(3):
    temp_idx = (pair_idx * 2) + 1
    c1 = plotted_component_pairs[pair_idx][0]
    c2 = plotted_component_pairs[pair_idx][1]
    # fit a KDE for these points
    kde = KernelDensity(bandwidth=0.5)
    kde.fit(np.hstack((t_samps[c1, :][:, None], t_samps[c2, :][:, None])))
    # some plotting params
    kde_lim_x = [-stats_t_train[pair_idx]['input']['var_c1'],
                 stats_t_train[pair_idx]['input']['var_c1']]
    kde_lim_y = [-stats_t_train[pair_idx]['input']['var_c2'],
                 stats_t_train[pair_idx]['input']['var_c2']]
    kde_sampling_density_x = 100
    kde_sampling_density_y = 100
    x = np.linspace(kde_lim_x[0], kde_lim_x[1], kde_sampling_density_x)
    y = np.linspace(kde_lim_y[0], kde_lim_y[1], kde_sampling_density_y)
    grid_points = np.array(list(itertools.product(x, y[::-1])))
    # generate plot
    ax = plt.subplot(3, 2, temp_idx)
    estimated_pdf = np.exp(kde.score_samples(grid_points))
    ax.imshow(estimated_pdf.reshape((kde_sampling_density_y,
                                     kde_sampling_density_x), order='F'),
              extent=[kde_lim_x[0], kde_lim_x[-1], kde_lim_y[0], kde_lim_y[-1]],
              aspect='auto')
    ax.text(kde_lim_x[0], kde_lim_y[0],
            format_string(stats_t_train[pair_idx], 'input'),
            horizontalalignment='left', verticalalignment='bottom',
            color='white', fontsize=7)
    ax.set_ylabel('Components {} & {}'.format(c1, c2))
    if pair_idx == 0:
      ax.set_title('Multivariate T (training_set)')

    # do the same thing for multivariate gaussian
    kde = KernelDensity(bandwidth=0.5)
    kde.fit(np.hstack((g_samps[c1, :][:, None], g_samps[c2, :][:, None])))
    # some plotting params
    kde_lim_x = [-stats_g[pair_idx]['var_c1'], stats_g[pair_idx]['var_c1']]
    kde_lim_y = [-stats_g[pair_idx]['var_c2'], stats_g[pair_idx]['var_c2']]
    kde_sampling_density_x = 100
    kde_sampling_density_y = 100
    x = np.linspace(kde_lim_x[0], kde_lim_x[1], kde_sampling_density_x)
    y = np.linspace(kde_lim_y[0], kde_lim_y[1], kde_sampling_density_y)
    grid_points = np.array(list(itertools.product(x, y[::-1])))
    # generate plot
    ax = plt.subplot(3, 2, temp_idx + 1)
    estimated_pdf = np.exp(kde.score_samples(grid_points))
    ax.imshow(estimated_pdf.reshape((kde_sampling_density_y,
                                     kde_sampling_density_x), order='F'),
              extent=[kde_lim_x[0], kde_lim_x[-1], kde_lim_y[0], kde_lim_y[1]],
              aspect='auto')
    ax.text(kde_lim_x[0], kde_lim_y[0], format_string(stats_g[pair_idx]),
            horizontalalignment='left', verticalalignment='bottom',
            color='white', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Multivariate Gaussian')
    plt.suptitle('2D Projections of multivariate samples from ' +
                 'two different distributions')


  # For the training data, plot some 2D projections of the data in original
  # input space, in the radially-gaussianized space, and then reconstructed
  # in the input space
  plt.figure(figsize=(15, 15))
  for pair_idx in range(3):
    temp_idx = (pair_idx * 3) + 1
    c1 = plotted_component_pairs[pair_idx][0]
    c2 = plotted_component_pairs[pair_idx][1]
    ax = plt.subplot(3, 3, temp_idx)
    ax.scatter(t_samps[c1, :], t_samps[c2, :], s=1)
    ax.set_ylabel('Components {} & {}'.format(c1, c2))
    ax.text(np.min(t_samps[c1]), np.min(t_samps[c2]),
            format_string(stats_t_train[pair_idx], 'input'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Samples in the input space')

    ax = plt.subplot(3, 3, temp_idx + 1)
    ax.scatter(rg_t_samps[c1, :], rg_t_samps[c2, :], s=1)
    ax.text(np.min(rg_t_samps[c1]), np.min(rg_t_samps[c2]),
            format_string(stats_t_train[pair_idx], 'rg'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Samples in the radially-gaussianized space')

    ax = plt.subplot(3, 3, temp_idx + 2)
    ax.scatter(reconstructed_t_samps[c1, :], reconstructed_t_samps[c2, :], s=1)
    ax.text(np.min(reconstructed_t_samps[c1]),
            np.min(reconstructed_t_samps[c2]),
            format_string(stats_t_train[pair_idx], 'rec_input'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Reconstructed samples after RG inversion')
  plt.suptitle('Samples from the general Student\'s T distributed training set')

  # Do the same thing for the testing dataset
  plt.figure(figsize=(15, 15))
  for pair_idx in range(3):
    temp_idx = (pair_idx * 3) + 1
    c1 = plotted_component_pairs[pair_idx][0]
    c2 = plotted_component_pairs[pair_idx][1]
    ax = plt.subplot(3, 3, temp_idx)
    ax.scatter(t_samps_test[c1, :], t_samps_test[c2, :], s=1)
    ax.set_ylabel('Components {} & {}'.format(c1, c2))
    ax.text(np.min(t_samps_test[c1]), np.min(t_samps_test[c2]),
            format_string(stats_t_test[pair_idx], 'input'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Samples in the input space')

    ax = plt.subplot(3, 3, temp_idx + 1)
    ax.scatter(rg_t_samps_test[c1, :], rg_t_samps_test[c2, :], s=1)
    ax.text(np.min(rg_t_samps_test[c1]), np.min(rg_t_samps_test[c2]),
            format_string(stats_t_test[pair_idx], 'rg'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Samples in the radially-gaussianized space')

    ax = plt.subplot(3, 3, temp_idx + 2)
    ax.scatter(reconstructed_t_samps_test[c1, :],
               reconstructed_t_samps_test[c2, :], s=1)
    ax.text(np.min(reconstructed_t_samps_test[c1]),
            np.min(reconstructed_t_samps_test[c2]),
            format_string(stats_t_test[pair_idx], 'rec_input'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Reconstructed samples after RG inversion')
  plt.suptitle('Samples from the general Student\'s T distributed testing set')

  # Finally, show the synthesized samples
  plt.figure(figsize=(15, 15))
  for pair_idx in range(3):
    temp_idx = (pair_idx * 3) + 1
    c1 = plotted_component_pairs[pair_idx][0]
    c2 = plotted_component_pairs[pair_idx][1]
    ax = plt.subplot(3, 3, temp_idx)
    ax.scatter(t_samps[c1, :], t_samps[c2, :], s=1)
    ax.set_ylabel('Components {} & {}'.format(c1, c2))
    ax.text(np.min(t_samps[c1]), np.min(t_samps[c2]),
            format_string(stats_t_train[pair_idx], 'input'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Training set samples in the input space')

    ax = plt.subplot(3, 3, temp_idx + 1)
    ax.scatter(synth_rg_samps[c1, :], synth_rg_samps[c2, :], s=1)
    ax.text(np.min(synth_rg_samps[c1]), np.min(synth_rg_samps[c2]),
            format_string(stats_synth[pair_idx], 'rg'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('Synthetic normally-distributed samples')

    ax = plt.subplot(3, 3, temp_idx + 2)
    ax.scatter(synth_samps[c1, :], synth_samps[c2, :], s=1)
    ax.text(np.min(synth_samps[c1]), np.min(synth_samps[c2]),
            format_string(stats_synth[pair_idx], 'rec_input'),
            horizontalalignment='left', verticalalignment='bottom',
            color='black', fontsize=7)
    if pair_idx == 0:
      ax.set_title('New, synthetic samples under the model')
  plt.suptitle('Generating synthetic data under the generative (inverse) model')

  plt.show()


def compute_var_kurt(data):
  """
  Helper script to compute variance and kurtosis for a 2d dataset
  """
  return {
    'var_c1': np.std(data[0, :])**2,
    'var_c2': np.std(data[1, :])**2,
    # we use the definition of kurtosis such that normal gives kurtosis of 3.0
    'kurt_c1': kurtosis(data[0, :], fisher=False),
    'kurt_c2': kurtosis(data[1, :], fisher=False)}


def format_string(stats_dict, key_label=None):
  """
  Helper to format a string we'll show at the base of plots
  """
  if key_label is None:
    return (
      'Var(c1): {:.1f}, '.format(stats_dict['var_c1']) +
      'Var(c2): {:.1f}, '.format(stats_dict['var_c2']) +
      'Kurtosis(c1): {:.1f}, '.format(stats_dict['kurt_c1']) +
      'Kurtosis(c2): {:.1f}'.format(stats_dict['kurt_c2']))
  else:
    return (
      'Var(c1): {:.1f}, '.format(stats_dict[key_label]['var_c1']) +
      'Var(c2): {:.1f}, '.format(stats_dict[key_label]['var_c2']) +
      'Kurtosis(c1): {:.1f}, '.format(stats_dict[key_label]['kurt_c1']) +
      'Kurtosis(c2): {:.1f}'.format(stats_dict[key_label]['kurt_c2']))


if __name__ == '__main__':
  run_rg()
