from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pickle

from constants import PLOT_COLORS

from gain import gain
from gan_utils import rmse_loss, binary_sampler
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

from fancyimpute import IterativeImputer
from fancyimpute import KNN

from scipy.stats import boxcox
from scipy.special import inv_boxcox
import numbers

def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, **kwargs):
    
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, s=1)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, lw=1)

# 10, 128, 0.9, 2500, 0.2
def main (alpha=1000, batch_size=128, hint_rate=0.9, 
  iterations=50000, miss_rate=0.2):
  
  gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'alpha': alpha,
                     'iterations': iterations}
  
  # Load data and introduce missingness
  #file_name = 'data/spam.csv'
  #data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  
  data_x = pickle.load(open('./missing_data.sav', 'rb'))
  data_x = data_x.transpose().astype(np.float)
  variables = ['urea', 'sodium', 'potassium', 
  'chloride', 'creatinine', 'total_calcium', 
  'corrected_total_calcium', 'magnesium', 
  'phosphate', 'total_bilirubin', 'ast', 
  'alanine_aminotransferase', 'alkaline_phosphatase', 
  'white_blood_cell_count', 'red_blood_cell_count', 
  'hemoglobin', 'mean_corpuscular_volume', 
  'mean_corpuscular_hemoglobin', 
  'mean_corpuscular_hemoglobin_concentration', 
  'platelet_count', 'lymphocyte_count', 
  'monocyte_count', 'neutrophil_count', 
  'eosinophil_count', 'basophil_count', 
  'c_reactive_protein', 'lactic_acid', 
  'lactate_dehydrogenase', 'albumin', 
  'mean_platelet_volume', 'glucose', 
  'bicarbonate', 'pco2', 'procalcitonin', 
  'hs_troponin_t', 'partial_thromboplastin_time', 
  'base_excess', 'osmolality']

  signed_variables = ['base_excess']
  no, dim = data_x.shape
  n_time_points = 5
  n_patients = int(no/n_time_points)

  if len(variables) != dim:
    print('Incompatible dimensions.')
    exit()

  scaling_parameters = []
  for d in range(0, dim):
    dim_values = data_x[:,d]
    nn_indices = np.isnan(dim_values) == False
    nn_dim_values = dim_values[nn_indices]
    delta = np.min(nn_dim_values)

    if variables[d] not in signed_variables and delta < 0:
      print('Negative value for unsigned variable: %s' % \
        variables[d])
      exit()

    nn_dim_values -= delta
    
    nn_dim_values, lmbda = boxcox(nn_dim_values+0.5)
    scaling_parameters.append([lmbda,delta])

    data_x[nn_indices,d] = nn_dim_values
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
  
  imputed_data_x_gan = gain(miss_data_x, gain_parameters)
  imputed_data_x = np.copy(imputed_data_x_gan)

  if False:
    n_gans = 5
    imputed_data_x_gans = []
    for x in range(0,n_gans):
      idxg = gain(miss_data_x, gain_parameters)
      idxg[data_m == 0] = np.nan
      imputed_data_x_gans.append(idxg)

    imputed_data_x_gan_all = np.concatenate(
      imputed_data_x_gans)

    idxga = np.copy(imputed_data_x_gan_all)
    
    # no longer sparse
    #normalizer = MinMaxScaler()
    #idxga = normalizer.fit_transform(idxga)
    from sklearn.metrics.pairwise import manhattan_distances
    imputer = KNNImputer(n_neighbors=9)

    for d in range(0, dim):
      for i in range(0, n_patients):
        if not np.isnan(data_x[i,d]) and \
          data_m[i,d] == 0:
          idxga[i,d] = np.nan

    imputed_data_x_knn = imputer. \
      fit_transform(idxga)
    #imputed_data_x_knn = normalizer. \
    #  inverse_transform(imputed_data_x_knn)

    imputed_data_x = np.copy(imputed_data_x_knn[:no,:])

  # imputer = IterativeImputer()
  #imputed_data_x_knn = imputer. \
  #  fit_transform(imputed_data_x_gan_all)
  transformed_arrs = []
  for arr in [data_x, miss_data_x, \
    imputed_data_x, imputed_data_x_gan]:
    for d in range(0, dim):
      lmbda, delta = scaling_parameters[d]
      nn_values = [x for x in arr[:,d] if x is not None]
      nn_values = inv_boxcox(nn_values, lmbda) - 0.5
      nn_values += delta
      k = 0
      for i in range(0, no):
        if arr[i,d] is not None:
          arr[i,d] = nn_values[k]
          k += 1
    transformed_arrs.append(arr)

  data_x, miss_data_x, \
    imputed_data_x, imputed_data_x_gan = transformed_arrs
  #imputed_data_x = imputed_data_x_knn[:no,:]

  distances_gan = [[] for d in range(0, dim)]
  distances = [[] for d in range(0, dim)]

  for i in range(0, n_patients):
    for j in range(0, dim):
      variable_name = variables[j]
      i_start = int(i*n_time_points)
      i_stop = int(i*n_time_points+n_time_points)
      
      orig_tuple = data_x[i_start:i_stop,j]
      corrupt_tuple = miss_data_x[i_start:i_stop,j]
      imput_tuple_gan = imputed_data_x_gan[i_start:i_stop,j]
      imput_tuple = imputed_data_x[i_start:i_stop,j]
      
      print(variable_name, orig_tuple, corrupt_tuple, imput_tuple)

      for k in range(0, n_time_points):
        a, b = orig_tuple[k], corrupt_tuple[k]
        c, d = imput_tuple_gan[k], imput_tuple[k]
        
        if not np.isnan(a) and np.isnan(b):
          distances_gan[j].append(c - a)
          distances[j].append(d - a)
  
  rrmses = []

  for j in range(0, dim):
    dists_gan = np.asarray(distances_gan[j])
    dists = np.asarray(distances[j])
    mbias_gan = np.round(np.mean(dists_gan), 2)
    mbias = np.round(np.mean(dists), 2)
    rmse = np.sqrt(np.mean(dists**2))
    dim_mean = np.mean([x for x \
      in data_x[:,j] if not np.isnan(x)])
    rrmse = np.round(rmse / dim_mean * 100, 2)
    rrmses.append(rrmse)

    print(variables[j], rrmse, '%', mbias_gan, mbias)

  n_fig_rows = 7
  n_fig_cols = 6

  n_fig_total = n_fig_rows * n_fig_cols

  if dim > n_fig_total:
    print('Warning: not all variables plotted')

  fig, axes = plt.subplots(\
    n_fig_rows, n_fig_cols, figsize=(15,15))
  top_title = 'Histogram of original data and imputed data'
  
  fig.suptitle(top_title, fontsize=8)

  for j in range(0, dim):
    ax_title = variables[j]
    ax = axes[int(j/n_fig_cols), j % n_fig_cols]
    ax.set_title(ax_title,fontdict={'fontsize':6})

    x_range = (
      np.min(imputed_data_x[:,j]), 
      np.max(imputed_data_x[:,j])
    )

    deleted_values = np.asarray([data_x[ii,j] \
      for ii in range(0, no) if \
      (not np.isnan(data_x[ii,j]) and \
       data_m[ii,j] == 0)])

    imputed_values = np.asarray([imputed_data_x[ii,j]
     for ii in range(0, no) if \
      (not np.isnan(data_x[ii,j]) and \
       data_m[ii,j] == 0)])

    imputed_values_gan = np.asarray([imputed_data_x_gan[ii,j]
     for ii in range(0, no) if \
      (not np.isnan(data_x[ii,j]) and \
       data_m[ii,j] == 0)])
    
    qqplot(deleted_values, imputed_values, ax=ax)
    continue

    try:
      kde_kws = {
        'shade': False, 'color': 'r',
        'bw':'scott', 
      }
      sns.distplot(imputed_values_gan, \
        kde_kws=kde_kws, hist=False,ax=ax)

      kde_kws = {
        'shade': False, 'color': '#000000',
        'bw':'scott', 
      }
      sns.distplot(deleted_values, \
        kde_kws=kde_kws, hist=False,ax=ax)

      #kde_kws = {
      #  'shade': False, 'color': PLOT_COLORS[0],
      #   'bw':'scott', 
      #}
      #sns.distplot(imputed_values, \
      #  kde_kws=kde_kws, hist=False,ax=ax)
    except:
      pass
    ax.set_ylabel('$p(x)$',fontsize=6)

  plt.setp(axes, yticks=[], xticks=[])
  plt.tight_layout(rect=[0,0.03,0,1.25])
  plt.subplots_adjust(hspace=1, wspace=0.35)
  plt.show()

  # Report the RMSE performance
  # rmse = rmse_loss(data_x, imputed_data_x, data_m)
  
  print()
  mrrmse = np.round(np.asarray(rrmses).mean(), 2)
  print('Average RMSE: ' + str(mrrmse) + '%')
  
  return imputed_data_x, rmse

imputed_data, rmse = main()
