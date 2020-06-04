from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
import pickle
import json

from constants import PLOT_COLORS

from gain import gain
from gan_utils import rmse_loss, binary_sampler
from sklearn.metrics import mean_squared_error


from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from scipy.stats import boxcox, t, sem, probplot, iqr, gaussian_kde, norm, bayes_mvs
from scipy.special import inv_boxcox
from scipy.optimize import fmin

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import load_model, model_from_json

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.ensemble import ExtraTreesRegressor
from fancyimpute import SimpleFill, IterativeSVD

import numbers
from math import sqrt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.markersize']=1
mpl.rcParams['lines.marker']='+'
mpl.rc('font', **{ 'family': ['Helvetica'] })

from sklearn.metrics import mutual_info_score as mutual_info

NUM_ITERATIONS = 3000
np.random.seed(10)

def plot_error_distributions(all_stats, fig, ax):
  
  kde_kws = { 'shade': False, 'bw':'scott' }
  
  for model_num, model_name in enumerate(['gan', 'knn', 'mice']):
    
    dist = [all_stats[variable_name][model_name]['rmse'] for variable_name in all_stats]
    sns.distplot(dist, hist=True, kde_kws={**{ 'color': PLOT_COLORS[model_num+1]}, **kde_kws}, ax=ax)
  
  fig.suptitle('Distribution of relative RMSEs for each variable, according to imputation method', fontsize=8)
  
  fig.tight_layout(rect=[0,0.03,0,1.25])
  fig.subplots_adjust(hspace=1, wspace=0.35)
  
def plot_distribution_residuals(output_arrays, ax):
  
  deleted_values, imputed_values_gan, imputed_values_knn, imputed_values_mice = output_arrays
  
  # Make QQ plot of observed vs. imputed values
  dist_max = np.max(np.concatenate([imputed_values_gan, deleted_values]))
  dist_min = np.min(np.concatenate([imputed_values_gan, deleted_values]))
  
  qqplot((deleted_values-dist_min) / dist_max, (imputed_values_gan-dist_min) / dist_max, ax=ax, color=PLOT_COLORS[1])
  qqplot((deleted_values-dist_min) / dist_max, (imputed_values_knn-dist_min) / dist_max, ax=ax, color=PLOT_COLORS[2])
  qqplot((deleted_values-dist_min) / dist_max, (imputed_values_mice-dist_min) / dist_max, ax=ax, color=PLOT_COLORS[3])
  
def plot_distribution_densities(output_arrays, ax):
  
  deleted_values, imputed_values_gan, imputed_values_knn, imputed_values_mice = output_arrays
  
  ax.set_xlabel('', fontsize=6)
  ax.set_ylabel('$p(x)$',fontsize=6)
  ax.set_yticks([])
  
  range_arrays = np.concatenate([deleted_values, imputed_values_gan])
  
  x_range = (np.min(range_arrays), 
    np.min([
      np.mean(range_arrays) + 3 * np.std(range_arrays), 
      np.max(range_arrays)
    ])
  )
  
  # Plot KDE for imputed distributions
  imputed_dists = [imputed_values_gan, imputed_values_knn, imputed_values_mice]
  imputed_dists_names = ['GAN', 'KNN', 'MICE']
  kde_kws = { 'shade': False, 'bw':'scott', 'clip': x_range }
  
  for model_num, imputed_dist in enumerate(imputed_dists):
    kws = {**{ 'color': PLOT_COLORS[model_num+1]}, **kde_kws}
    if model_num == 1: kws['alpha'] = 0.75
    lab = imputed_dists_names[model_num]
    sns.distplot(imputed_dist, hist=False, kde_kws=kws, ax=ax, label=lab)
  
  # Plot KDE for missing data held out
  sns.distplot(deleted_values, hist=True, kde=True, 
    kde_kws={**{ 'color': '#000000', 'alpha': 0.25}, **kde_kws},
    hist_kws={ 'color': PLOT_COLORS[0], 'alpha': 0.25},
    ax=ax, label='Observed')  
  
  ax.legend(fontsize=6)

def plot_distribution_summaries(output_arrays, ax):
  
  m = 1
  low_lim, high_lim = [], []
  
  for output_array in output_arrays:
    
    mean, var, std = bayes_mvs(output_array)
    
    low_lim.append(mean[0] - 1.96 * std[0])
    high_lim.append(mean[0] + 1.96 * std[0])
    
    markers, caps, bars = ax.errorbar(
      [m-0.5], [mean[0]], yerr=[std[0]*1.96], fmt='o', c='#000000', 
      capsize=3, elinewidth=1, markeredgewidth=1)

    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    
    ax.scatter([m for _ in range(len(output_array))], output_array, 
      marker='+', s=1, c=PLOT_COLORS[m-1], alpha=0.25)
    
    m += 1
  
  low_lim = np.max([np.min(low_lim), 0])
  high_lim = np.max(high_lim)
  span_lim = high_lim - low_lim
  
  low_lim = low_lim - 2 * span_lim / 10
  high_lim = high_lim + 3 * span_lim / 10

  ax.set_ylim(low_lim, high_lim)
  ax.set_xticks([m+0.75 for m in range(len(output_arrays))])
  ax.set_xticklabels(['Obs.', 'GAN', 'KNN', 'MICE'])
  
  [lab.set_fontsize(6.5) for lab in ax.get_xticklabels()]
  [lab.set_fontsize(6.5) for lab in ax.get_yticklabels()]
  
  m = 1
  for output_array in output_arrays:
    
    mean, var, std = bayes_mvs(output_array)
    y1, y2 = ax.get_ylim()
    ax.annotate('%.2f ± %.2f' % (mean[0], std[0]), 
      xy=(m-0.5, mean[0] + 1.96 * std[0] + (y2-y1) / 10), fontsize=6)
    m += 1
  
def qqplot_1sample(x, ax=None, color=None, draw_line=True):

  probplot(x, dist="norm",plot=ax)
  
def qqplot(x, y, ax=None, color=None, draw_line=True):
    
    if ax is None: ax = plt.gca()
    
    quantiles = min(len(x), len(y))
    quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    
    x_quantiles = np.quantile(x, quantiles, interpolation='nearest')
    y_quantiles = np.quantile(y, quantiles, interpolation='nearest')
    
    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, s=1, alpha=0.5)
    x = np.linspace(0, 1.0, 50)
    ax.plot(x, x, lw=1,color='gray',alpha=0.5)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    
    lim = np.max([
      np.min([np.max(x),1.0]),
      np.min([np.max(y),1.0])
    ])

    ax.set_ylabel('$quantiles$',fontsize=6)
    
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

variables = [
  'urea', 'sodium', 'potassium', 'chloride', 'creatinine', 
  'total_calcium', 'corrected_total_calcium', 'magnesium', 
  'phosphate', 'total_bilirubin', 'ast', 'alanine_aminotransferase', 
  'alkaline_phosphatase', 'white_blood_cell_count', 'hemoglobin', 
  'platelet_count', 'lymphocyte_count', 'monocyte_count', 
  'neutrophil_count', 'c_reactive_protein', 'lactic_acid', 'albumin', 
  'mean_platelet_volume', 'glucose', 'hs_troponin_t', 
  'partial_thromboplastin_time', 'bicarbonate', 'anion_gap', 'pco2', 
  'procalcitonin', 'base_excess', 'osmolality', 'lipase']

remove_variables = []#['ast', 'procalcitonin', 'lipase', 'base_excess', 'lymphocyte_count']

def main (iterations=NUM_ITERATIONS, batch_size=128, hint_rate=0.5, miss_rate=0.3):
  
  gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'iterations': iterations}
  
  enable_transform = False
  remove_outliers = False
  n_time_points = 3
  
  data_x = pickle.load(open('./missing_data.sav', 'rb'))
  data_x = data_x.transpose().astype(np.float)[:,:]
  
  # Remove variables with more 
  no, dim = data_x.shape
  removed = 0
  for d in range(0,dim):
    if variables[d-removed] in remove_variables:
      variables.remove(variables[d-removed])
      data_x = np.delete(data_x, d-removed, axis=1)
      removed += 1
    
  no, dim = data_x.shape
  
  if len(variables) != dim:
    print(len(variables), dim)
    print('Incompatible dimensions.')
    exit()
  
  no_total = no * dim
  no_nan = np.count_nonzero(np.isnan(data_x.flatten()) == True)
  no_not_nan = no_total - no_nan
  n_patients = int(no/n_time_points)
  
  miss_data_x = np.copy(data_x)
  
  print('Input shape', no, 'x', dim)
  print('NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x[data_m == 0] = np.nan
  
  #transformer = RobustScaler()
  #miss_data_x = transformer.fit_transform(miss_data_x)
  
  no_nan = np.count_nonzero(np.isnan(miss_data_x.flatten()) == True)
  no_not_nan = no_total - no_nan

  print('After removal, NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))
  
  real_miss_rate = (no_nan / no_total * 100)
  
  miss_data_x_gan_tmp = np.zeros((n_patients,dim*n_time_points))
  
  # Swap (one row per time point) to (one column per time point)
  for i in range(0, n_patients):
    for j in range(0, dim):
      for n in range(0, n_time_points):
        miss_data_x_gan_tmp[i,n*dim+j] = miss_data_x[i*n_time_points+n,j] 
  
  imputed_data_x_gan_tmp = gain(
    miss_data_x_gan_tmp, gain_parameters)
  
  imputed_data_x_gan = np.copy(miss_data_x)
  
  ## Swap (one column per time point) to (one row per time point)
  for i in range(0, n_patients):
    for j in range(0, dim):
      for n in range(0, n_time_points):
        imputed_data_x_gan[i*n_time_points+n,j] = imputed_data_x_gan_tmp[i,n*dim+j]
  
  imputer = KNNImputer(n_neighbors=5)
  imputed_data_x_knn = imputer.fit_transform(miss_data_x)
  
  imputer = IterativeImputer(verbose=True)
  imputed_data_x_mice = imputer.fit_transform(miss_data_x)
  
  #imputed_data_x_gan = transformer.inverse_transform(imputed_data_x_gan)
  #imputed_data_x_knn = transformer.inverse_transform(imputed_data_x_knn)
  #imputed_data_x_mice = transformer.inverse_transform(imputed_data_x_mice)
  
  # Save imputed data to disk
  pickle.dump(imputed_data_x_gan,open('./filled_data.sav', 'wb'))
  
  # Get residuals for computation of stats
  distances_gan = np.zeros((dim, n_time_points*n_patients))
  distances_knn = np.zeros((dim, n_time_points*n_patients))
  distances_mice = np.zeros((dim, n_time_points*n_patients))
  distributions = { 'deleted': [], 'gan': [], 'knn': [], 'mice': []}
  
  from scipy.stats import iqr
  
  for j in range(0, dim):
    
    nn_values = data_x[:,j].flatten()
    nn_values = nn_values[~np.isnan(nn_values)]

    dim_iqr = np.mean(nn_values) # iqr(nn_values)
    
    for i in range(0, n_patients):
      variable_name = variables[j]
      i_start = int(i*n_time_points)
      i_stop = int(i*n_time_points+n_time_points)
      
      original_tuple = data_x[i_start:i_stop,j]
      corrupted_tuple = miss_data_x[i_start:i_stop,j]
      imputed_tuple_gan = imputed_data_x_gan[i_start:i_stop,j]
      imputed_tuple_knn = imputed_data_x_knn[i_start:i_stop,j]
      imputed_tuple_mice = imputed_data_x_mice[i_start:i_stop,j]
      
      #if i == 1 or i == 2:
      #  print(original_tuple, corrupted_tuple, imputed_tuple_gan, imputed_tuple_knn)
      
      for k in range(0, n_time_points):
        a, b, c, d = original_tuple[k], imputed_tuple_gan[k], \
                     imputed_tuple_knn[k], imputed_tuple_mice[k]
        if np.isnan(a) or data_m[i_start+k,j] != 0: continue
        #if i % 10 == 0: print(variable_name, a,b,c,d, b-a)
        distances_gan[j,i*k] = (b - a)
        distances_knn[j,i*k] = (c - a) 
        distances_mice[j,i*k] = (d - a)

  # Compute distance statistics
  stats = { 'gan': {}, 'knn': {}, 'mice': {} }
  all_stats = {}
  
  for j in range(0, dim):
    
    print('%d. Imputed variable: %s' % (j, variables[j]))
    
    current_stats = dict(stats) # make a copy
    
    # Stats for original data
    dim_mean = np.mean([x for x in data_x[:,j] if not np.isnan(x)])
    dim_max = np.max([x for x in data_x[:,j] if not np.isnan(x)])
    
    # Stats for GAN
    current_stats['gan']['rmse'] = np.sqrt(np.mean(distances_gan[j]**2))
    current_stats['gan']['mape'] = np.mean(np.abs(distances_gan[j]))
    
    # Stats for KNN
    current_stats['knn']['rmse'] = np.sqrt(np.mean(distances_knn[j]**2))
    current_stats['knn']['mape'] = np.mean(np.abs(distances_knn[j]))
    
    # Stats for MICE
    current_stats['mice']['rmse'] = np.sqrt(np.mean(distances_mice[j]**2))
    current_stats['mice']['mape'] = np.mean(np.abs(distances_mice[j]))
    
    for model_name in current_stats:
      model = stats[model_name]
      print('... %s - RMSE: %.3f, MAPE: %.3f' % \
        (model_name, model['rmse'], model['mape']))
    
    all_stats[variables[j]] = current_stats
    
    print()
    
  n_fig_rows, n_fig_cols = 6, 6
  n_fig_total = n_fig_rows * n_fig_cols

  if dim > n_fig_total: print('Warning: not all variables plotted')

  all_fig_axes = [plt.subplots(n_fig_rows, n_fig_cols, figsize=(15,15)) for _ in range(0,4)]

  for j in range(0, dim):

    dim_not_nan = np.count_nonzero(~np.isnan(data_x[:,j]))
    deleted_no = np.count_nonzero(np.isnan(miss_data_x[:,j]) & ~np.isnan(data_x[:,j]))
    ax_title = variables[j] + (' (%d of %d observed)' % (deleted_no, dim_not_nan))
    
    dim_axes = [fig_axes[1][int(j/n_fig_cols), j % n_fig_cols] for fig_axes in all_fig_axes]
        
    [ax.set_title(ax_title,fontdict={'fontsize':7, 'fontweight': 'bold'}) for ax in dim_axes]
    
    input_arrays = [data_x, imputed_data_x_gan, imputed_data_x_knn, imputed_data_x_mice]
    
    output_arrays = [
      np.asarray([input_arr[ii,j] for ii in range(0, no) if \
        (not np.isnan(data_x[ii,j]) and \
        data_m[ii,j] == 0)]) for input_arr in input_arrays
    ]
  
    deleted_values, imputed_values_gan, imputed_values_knn, imputed_values_mice = output_arrays
    
    plot_distribution_densities(output_arrays, dim_axes[0])
    plot_distribution_residuals(output_arrays, dim_axes[1])
    plot_distribution_summaries(output_arrays, dim_axes[2])
    
    # Make QQ plot of original and deleted values vs. normal distribution
    #dist_max = np.max(np.concatenate([imputed_values_gan, deleted_values]))
    #qqplot_1sample((data_x[~np.isnan(data_x[:,j]),j] - dist_min) / dist_max, ax=ax3, color='b')
    #qqplot_1sample((data_x[data_m[:,j] == 0,j] - dist_min) / dist_max, ax=ax3, color='r',draw_line=False)
    
  # Figure 1
  fig1 = all_fig_axes[0][0]
  top_title = 'KDE plot of original data (black) and data imputed using GAN (red) and KNN (blue)'
  fig1.suptitle(top_title, fontsize=8)

  fig1.tight_layout(rect=[0,0.03,0,1.25])
  fig1.subplots_adjust(hspace=1, wspace=0.35)

  # Figure 2
  fig2 = all_fig_axes[1][0]
  top_title = 'Q-Q plot of erased vs. predicted values, for each imputation method'
  fig2.suptitle(top_title, fontsize=8)

  fig2.tight_layout(rect=[0,0.03,0,1.25])
  fig2.subplots_adjust(hspace=1, wspace=0.35)
  
  # Figure 3
  fig3 = all_fig_axes[2][0]
  top_title = 'Bayesian confidence intervals for the mean, var, and standard deviation, for imputed and observed values'
  fig3.suptitle(top_title, fontsize=8)

  fig3.tight_layout(rect=[0,0.03,0,1.25])
  fig3.subplots_adjust(hspace=1, wspace=0.35)
  fig3.legend(labels=['Observed', 'GAN', 'KNN', 'MICE'])
  
  # Figure 4
  fig5, ax5 = plt.subplots(1,1)
  plot_error_distributions(all_stats, fig5, ax5)
  
  plt.show()
  
  for model_name in ['gan', 'knn', 'mice']:
    rrmses = [all_stats[variable_name][model_name]['rmse'] for variable_name in all_stats]
    mrrmse = np.round(np.asarray(rrmses).mean(), 2)
    print('Average RMSE (%s): ' % model_name, mrrmse)

  return all_stats

for k in np.linspace(0.2,0.8, 6):
  print('----------')
  stats = main(miss_rate = k)
  print(stats)
