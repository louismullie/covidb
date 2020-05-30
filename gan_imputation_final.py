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

import seaborn as sns
import matplotlib as mpl
mpl.rcParams['lines.markersize']=1
mpl.rcParams['lines.marker']='+'
mpl.rc('font', **{ 'family': ['Helvetica'] })
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from scipy.stats import boxcox, t, sem, probplot
from scipy.special import inv_boxcox

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
from scipy.stats import norm 

np.random.seed(1)
def qqplot_1sample(x, ax=None, color=None, draw_line=True):

  probplot(x, dist="norm",plot=ax)
  
#def qqplot_sample_subsample(x_o, y_o, ax=None, color=None, draw_line=True):
#  probplot_arrays, lsf = probplot(x_o, dist="norm")
#  x, y = probplot_arrays
#  ax.scatter(x,y,s=1,c='b',marker='+',alpha=0.5)
#  
#  x2 = np.linspace(np.min(x), np.max(y), 50)
#  y2 = np.poly1d(np.polyfit(x,y, 1))(x2)
#  
#  if draw_line: ax.plot(x2,y2, lw=1,color='gray',alpha=0.5)
#  
#  ax.set_xticks([-2.5, -1, 0, 1, 2.5])
#  ax.set_yticks([0, 0.5, 1.0])
#  #ax.set_xlim(-2.5, 2.5)
#  #ax.set_ylim(0, 1.0)

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

encoders = []    
      
for i in range(0, len(variables)):
    variable = variables[i]
    with open('./models/autoencoders/%s.json' % variable,'r') as f:
      model_json = f.read()
    encoder_model = model_from_json(model_json)
    encoder_model.load_weights('./models/autoencoders/%s.h5' % variable)

    encoders.append(encoder_model)
    
def main (alpha=1000, batch_size=128, hint_rate=0.05, 
  iterations=5000, miss_rate=0.3):
  
  gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'alpha': alpha,
                     'iterations': iterations}
  
  enable_transform = False
  remove_outliers = False
  n_time_points = 3
  
  data_x = pickle.load(open('./missing_data.sav', 'rb'))
  data_x = data_x.transpose().astype(np.float)[:,:]

  no, dim = data_x.shape
  
  if len(variables) != dim:
    print(len(variables), dim)
    print('Incompatible dimensions.')
    exit()
  
  no_total = no * dim
  no_nan = np.count_nonzero(np.isnan(data_x.flatten()) == True)
  no_not_nan = no_total - no_nan
  n_patients = int(no/n_time_points)
  
  data_x_encoded = np.copy(data_x)
  miss_data_x = np.copy(data_x)
  miss_data_x_enc = np.copy(data_x)
  
  scalers = []
  
  print('Input shape', no, 'x', dim)
  print('NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))
  
  for i in range(0, dim):
    variable, var_x = variables[i], np.copy(data_x[:,i])
    encoder_model = encoders[i]
    
    nn_indices = ~np.isnan(data_x_encoded[:,i])
    nn_values = data_x[:,i][nn_indices]

    scaler = MinMaxScaler()
    var_x_scaled = scaler.fit_transform(var_x.reshape((-1,1)))
    
    enc_x_scaled = encoder_model.predict(var_x_scaled)
    enc_x_unscaled = scaler.inverse_transform(enc_x_scaled)
    data_x_encoded[:,i] = enc_x_unscaled.flatten()
    
    scalers.append(scaler)
      
    #print(var_x, '----', enc_x_scaled, '----', enc_x_unscaled.flatten())
    print('Loaded model for %s...' % variable)
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  
  miss_data_x[data_m == 0] = np.nan
  miss_data_x_enc[data_m == 0] = np.nan

  no_nan = np.count_nonzero(np.isnan(miss_data_x.flatten()) == True)
  no_not_nan = no_total - no_nan

  print('After removal, NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))
  
  real_miss_rate = (no_nan / no_total * 100)
  
  transformer = None
  
  if enable_transform:  
    print('Applying transformation...')
    transformer1 = RobustScaler()
    #transformer2 = PowerTransformer()
    miss_data_x_enc = transformer1.fit_transform(miss_data_x)
    miss_data_x_enc[data_m == 0] = np.nan
  
  imputed_data_x_gan = gain(
    miss_data_x_enc, gain_parameters)
  
  if enable_transform:  
    print('Reversing transformation...')
    imputed_data_x_gan = transformer1.inverse_transform(imputed_data_x_gan)
  
  imputer = KNNImputer(n_neighbors=5)
  imputed_data_x_knn = imputer.fit_transform(miss_data_x)
  
  imputer = IterativeImputer()
  imputed_data_x_mice = imputer.fit_transform(miss_data_x)
  
  # Save imputed data to disk
  pickle.dump(imputed_data_x_gan,open('./filled_data.sav', 'wb'))
  
  # Get residuals for computation of stats
  distances_gan = np.zeros((dim, n_time_points*n_patients))
  distances_knn = np.zeros((dim, n_time_points*n_patients))
  distances_mice = np.zeros((dim, n_time_points*n_patients))

  for i in range(0, n_patients):
    for j in range(0, dim):
      variable_name = variables[j]
      i_start = int(i*n_time_points)
      i_stop = int(i*n_time_points+n_time_points)
      
      original_tuple = data_x[i_start:i_stop,j]
      corrupted_tuple = miss_data_x[i_start:i_stop,j]
      imputed_tuple_gan = imputed_data_x_gan[i_start:i_stop,j]
      imputed_tuple_knn = imputed_data_x_knn[i_start:i_stop,j]
      imputed_tuple_mice = imputed_data_x_mice[i_start:i_stop,j]
      
      if i == 1 or i == 2:
        print(original_tuple, corrupted_tuple, imputed_tuple_gan, imputed_tuple_knn)
      for k in range(0, n_time_points):
        a, b, c, d = original_tuple[k], imputed_tuple_gan[k], imputed_tuple_knn[k], imputed_tuple_mice[k]
        if np.isnan(a) or data_m[i_start+k,j] != 0: continue
        if i % 10 == 0: print(variable_name, a,b,c,d, b-a)
        distances_gan[j,i*k] = (b - a)
        distances_knn[j,i*k] = c - a
        distances_mice[j,i*k] = d - a
  
  # Compute distance statistics
  rrmses_gan, mean_biases, median_biases, bias_cis = [], [], [], []
  rrmses_knn, mean_biases_knn, median_biases_knn, bias_cis_knn = [], [], [], []
  rrmses_mice = []

  for j in range(0, dim):
    
    # Stats for original data
    dim_mean = np.mean([x for x in data_x[:,j] if not np.isnan(x)])
    dim_max = np.max([x for x in data_x[:,j] if not np.isnan(x)])
    
    dists_gan = distances_gan[j]
    dists_knn = distances_knn[j]
    dists_mice = distances_mice[j]
    
    # Stats for GAN
    mean_bias = np.round(np.mean(dists_gan), 4)
    median_bias = np.round(np.median(dists_gan), 4)
    mean_ci_95 = mean_confidence_interval(dists_gan)
    rmse_gan = np.sqrt(np.mean(dists_gan**2))
    rrmse_gan = rmse_gan / dim_mean
    
    bias_cis.append([mean_ci_95[1], mean_ci_95[2]])
    mean_biases.append(mean_bias)
    median_biases.append(median_bias)
    rrmses_gan.append(rrmse_gan)
    
    # Stats for KNN
    rmse_knn = np.sqrt(np.mean(dists_knn**2))
    rrmse_knn = rmse_knn / dim_mean
    rrmses_knn.append(rrmse_knn)
    
    # Stats for MICE
    rmse_mice = np.sqrt(np.mean(dists_mice**2))
    rrmse_mice = rmse_mice / dim_mean
    rrmses_mice.append(rrmse_mice)
    
    print(variables[j], ' - rrmse: ', rmse_gan, 'median bias: %.2f' % median_bias,
      '%%, bias: %.2f (95%% CI, %.2f to %.2f)' % mean_ci_95)

  n_fig_rows = 6
  n_fig_cols = 6

  n_fig_total = n_fig_rows * n_fig_cols

  if dim > n_fig_total:
    print('Warning: not all variables plotted')

  fig, axes = plt.subplots(\
    n_fig_rows, n_fig_cols, figsize=(15,15))
  fig2, axes2 = plt.subplots(\
    n_fig_rows, n_fig_cols, figsize=(15,15))
  fig3, axes3 = plt.subplots(\
    n_fig_rows, n_fig_cols, figsize=(15,15))

  for j in range(0, dim):

    dim_not_nan = np.count_nonzero(~np.isnan(data_x[:,j]))
    ax_title = variables[j] + (' (n=%d)' % dim_not_nan)
    
    ax = axes[int(j/n_fig_cols), j % n_fig_cols]
    ax2 = axes2[int(j/n_fig_cols), j % n_fig_cols]
    ax3 = axes3[int(j/n_fig_cols), j % n_fig_cols]
    
    input_arrays = [data_x, imputed_data_x_gan, imputed_data_x_knn, imputed_data_x_mice]
    
    output_arrays = [
      np.asarray([input_arr[ii,j] for ii in range(0, no) if \
        (not np.isnan(data_x[ii,j]) and \
        data_m[ii,j] == 0)]) for input_arr in input_arrays
    ]
    
    deleted_values, imputed_values_gan, imputed_values_knn, imputed_values_mice = output_arrays
    
    # Make KDE
    low_ci, high_ci = bias_cis[j]
    xlabel = 'mean bias = %.2f (95%% CI, %.2f to %.2f)' % \
      (mean_biases[j], low_ci, high_ci)
      
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel('$p(x)$',fontsize=6)
    ax.set_yticks([])
    
    range_arrays = np.concatenate([deleted_values, imputed_values_gan])
    
    x_range = (np.min(range_arrays), 
      np.min([
        np.mean(range_arrays) + 3 * np.std(range_arrays), 
        np.max(range_arrays)
      ])
    )
    
    all_values_gan = (imputed_values_gan * 2 + imputed_values_mice * 1) / 3
    
    
    
    kde_kws = { 'shade': False, 'bw':'scott', 'clip': x_range }
    
    sns.distplot(all_values_gan, hist=False,
      kde_kws={**{ 'color': 'r'}, **kde_kws}, ax=ax)
      
    #sns.distplot(imputed_values_gan, hist=False,
    #  kde_kws={**{ 'color': 'r'}, **kde_kws}, ax=ax)
    #
    #sns.distplot(imputed_values_knn, hist=False,
    #  kde_kws={**{ 'color': 'b', 'alpha': 0.5 }, **kde_kws},ax=ax)
    #
    #sns.distplot(imputed_values_mice, hist=False,
    #  kde_kws={**{ 'color': 'g', 'alpha': 0.5 }, **kde_kws},ax=ax)
    #
    sns.distplot(deleted_values, hist=False,
      kde_kws={**{ 'color': '#000000'}, **kde_kws},ax=ax)

    # Make QQ plot of observed vs. imputed values
    dist_max = np.max(np.concatenate([imputed_values_gan, deleted_values]))
    dist_min = np.min(np.concatenate([imputed_values_gan, deleted_values]))
    
    qqplot((deleted_values-dist_min) / dist_max, (imputed_values_gan-dist_min) / dist_max, ax=ax2, color='r')
    qqplot((deleted_values-dist_min) / dist_max, (imputed_values_knn-dist_min) / dist_max, ax=ax2, color='b')
    qqplot((deleted_values-dist_min) / dist_max, (imputed_values_mice-dist_min) / dist_max, ax=ax2, color='g')
    
    # Make QQ plot of original and deleted values vs. normal distribution
    dist_max = np.max(np.concatenate([imputed_values_gan, deleted_values]))
    
    sample_indices = []
    
    qqplot_1sample((data_x[~np.isnan(data_x[:,j]),j] - dist_min) / dist_max, ax=ax3, color='b')
    #qqplot_1sample((data_x[data_m[:,j] == 0,j] - dist_min) / dist_max, ax=ax3, color='r',draw_line=False)
    
    ax.set_title(ax_title,fontdict={'fontsize':6})
    ax2.set_title(ax_title,fontdict={'fontsize':6})
    ax3.set_title(ax_title,fontdict={'fontsize':6})
    
  top_title = 'KDE plot of original data (black) and data imputed using GAN (red) and KNN (blue)'
  fig.suptitle(top_title, fontsize=8)
  fig.legend(labels=['GAN', 'KNN', 'MICE', 'Observed'])

  fig.tight_layout(rect=[0,0.03,0,1.25])
  fig.subplots_adjust(hspace=1, wspace=0.35)

  top_title = 'Q-Q plot of observed vs. predicted values'
  fig2.suptitle(top_title, fontsize=8)

  fig2.tight_layout(rect=[0,0.03,0,1.25])
  fig2.subplots_adjust(hspace=1, wspace=0.35)

  top_title = 'Q-Q plot for distribution of observed values (black) and deleted values (red)'
  fig3.suptitle(top_title, fontsize=8)

  fig3.tight_layout(rect=[0,0.03,0,1.25])
  fig3.subplots_adjust(hspace=1, wspace=0.35)
  
  fig4, ax4 = plt.subplots(1,1)
  
  kde_kws = { 'shade': False, 'bw':'scott' }
  
  sns.distplot(rrmses_gan, hist=True,
    kde_kws={**{ 'color': 'r'}, **kde_kws}, ax=ax4)
  
  sns.distplot(rrmses_knn, hist=True,
    kde_kws={**{ 'color': 'b'}, **kde_kws}, ax=ax4)
    
  sns.distplot(rrmses_mice, hist=True,
    kde_kws={**{ 'color': 'g'}, **kde_kws}, ax=ax4)
  
  fig4.suptitle('Distribution of relative RMSEs for each variable, according to imputation method', fontsize=8)
  
  fig4.tight_layout(rect=[0,0.03,0,1.25])
  fig4.subplots_adjust(hspace=1, wspace=0.35)
  
  plt.show()
  
  print()
  mrrmse_gan = np.round(np.asarray(rrmses_gan).mean(), 2)
  print('Average RMSE (GAN): ', mrrmse_gan)
  #print('Average STD (GAN): ', std_gan, '%')

  print()
  mrrmse_knn = np.round(np.asarray(rrmses_knn).mean(), 2)
  print('Average RMSE (KNN): ', mrrmse_knn)
  #print('Average STD (KNN): ', std_knn, '%')

  print()
  mrrmse_mice = np.round(np.asarray(rrmses_mice).mean(), 2)
  print('Average RMSE (MICE): ', mrrmse_mice)
  #print('Average STD (MICE): ', std_mice, '%')
  
  return real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice

errors = []
for k in np.linspace(0.2,0.8, 6):
  print('----------')
  real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice = main(miss_rate = 0.2)
  errors.append([real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice])
  print(real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice)
  
print(errors)