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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox, t, sem
from scipy.special import inv_boxcox

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

import numbers

import pandas as pd

def find_outliers(variable_name, values):
  RANDOM_SEED = 101

  X_train = pd.DataFrame(values.reshape(-1, 1))
  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)

  # No of Neurons in each Layer [9,6,3,2,3,6,9]
  input_dim = X_train.shape[1]
  encoding_dim = 6

  input_layer = Input(shape=(input_dim, ))
  encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
  encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
  encoder = Dense(int(2), activation="tanh")(encoder)
  decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
  decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
  decoder = Dense(input_dim, activation='tanh')(decoder)
  autoencoder = Model(inputs=input_layer, outputs=decoder)
  #autoencoder.summary()

  nb_epoch = 100
  batch_size = 32
  autoencoder.compile(optimizer='adam', loss='mse' )

  history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=nb_epoch,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.1,
    verbose=False
  )

  with open('./models/autoencoders/%s.json' % \
    variable_name, 'w') as json_file:
    json_file.write(autoencoder.to_json())

  autoencoder.save_weights('./models/autoencoders/%s.h5' % \
    variable_name, 'w')

  df_history = pd.DataFrame(history.history)
  predictions = autoencoder.predict(X_train_scaled)

  mse = np.mean(np.power(X_train_scaled - predictions, 2),axis=1)
  outlier_indices = mse > 0.025
  
  print(variable_name, values[outlier_indices])
  #plt.title(variable_name); plt.plot(mse); plt.show()
  return outlier_indices

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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# 10, 128, 0.9, 2500, 0.2
def main (alpha=100, batch_size=16, hint_rate=0.9, 
  iterations=1500, miss_rate=0.2):
  
  gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'alpha': alpha,
                     'iterations': iterations}
  
  ENABLE_BOXCOX = True

  # Load data and introduce missingness
  #file_name = 'data/spam.csv'
  #data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  
  remove_outliers = False

  if remove_outliers:
    data_x = pickle.load(open('./missing_data.sav', 'rb'))
    data_x = data_x.transpose().astype(np.float)
  else:
    data_x = pickle.load(open('./denoised_missing_data.sav', 'rb')) 

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

  signed_variables = ['base_excess']
  no, dim = data_x.shape

  no_total = no * dim
  no_nan = np.count_nonzero(np.isnan(data_x.flatten()) == True)
  no_not_nan = no_total - no_nan
  print('Input shape', no, 'x', dim)
  print('NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))

  n_time_points = 3
  n_patients = int(no/n_time_points)

  if len(variables) != dim:
    print(len(variables), dim)
    print('Incompatible dimensions.')
    exit()

  # Outlier detection using autoencoder
  if remove_outliers:

    for d in range(0, dim):
      dim_values = data_x[:,d]
      nn_indices = np.isnan(dim_values) == False
      nn_dim_values = dim_values[nn_indices]
      out_indices = find_outliers(variables[d], nn_dim_values)
      nn_dim_values[out_indices] = np.nan
      dim_values[nn_indices] = nn_dim_values
      data_x[:,d] = dim_values

    pickle.dump(data_x, open('./denoised_missing_data.sav', 'wb'))
    

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

    if ENABLE_BOXCOX:
      nn_dim_values -= delta
    
      nn_dim_values, lmbda = boxcox(nn_dim_values+0.5)
      scaling_parameters.append([lmbda,delta])

      data_x[nn_indices,d] = nn_dim_values
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)

  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
  
  no_nan = np.count_nonzero(np.isnan(miss_data_x.flatten()) == True)
  no_not_nan = no_total - no_nan

  print('After removal, NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))
  
  from sklearn.ensemble import ExtraTreesRegressor
  from fancyimpute import SimpleFill, IterativeSVD
  imputed_data_x_gan = gain(miss_data_x, gain_parameters)
  imputed_data_x = np.copy(imputed_data_x_gan)

  #imputer = IterativeSVD()#IterativeImputer(max_iter=2, verbose=True)
  #imputed_data_x_svd = imputer.fit_transform(miss_data_x)
  imputed_data_x_svd = np.copy(imputed_data_x_gan)

  imputer = KNNImputer()
  imputed_data_x_knn = imputer.fit_transform(miss_data_x)

  transformed_arrs = []
  for arr in [data_x, miss_data_x, imputed_data_x, \
    imputed_data_x_gan, imputed_data_x_knn, imputed_data_x_svd]:
    for d in range(0, dim):
      
      nn_values = [x for x in arr[:,d] if x is not None]

      if ENABLE_BOXCOX:
        lmbda, delta = scaling_parameters[d]
        nn_values = inv_boxcox(nn_values, lmbda) - 0.5
        nn_values += delta

      k = 0
      for i in range(0, no):
        if arr[i,d] is not None:
          arr[i,d] = nn_values[k]
          k += 1
    transformed_arrs.append(arr)

  data_x, miss_data_x, imputed_data_x, \
    imputed_data_x_gan, imputed_data_x_knn, \
    imputed_data_x_svd = transformed_arrs
  
  pickle.dump(imputed_data_x,open('./filled_data.sav', 'wb'))
  
  distances_gan = [[] for d in range(0, dim)]
  distances_knn = [[] for d in range(0, dim)]

  for i in range(0, n_patients):
    for j in range(0, dim):
      variable_name = variables[j]
      i_start = int(i*n_time_points)
      i_stop = int(i*n_time_points+n_time_points)
      
      orig_tuple = data_x[i_start:i_stop,j]
      corrupt_tuple = miss_data_x[i_start:i_stop,j]
      imput_tuple_gan = imputed_data_x_gan[i_start:i_stop,j]
      imput_tuple = imputed_data_x[i_start:i_stop,j]
      
      #print(variable_name, orig_tuple, corrupt_tuple, imput_tuple)

      for k in range(0, n_time_points):
        a, b = orig_tuple[k], corrupt_tuple[k]
        c, d = imput_tuple_gan[k], imput_tuple[k]
        
        if not np.isnan(a) and np.isnan(b):
          distances_gan[j].append(c - a)
          distances_knn[j].append(d - a)
  
  rrmses = []
  mbiases = []
  cis = []

  for j in range(0, dim):
    dists_gan = np.asarray(distances_gan[j])
    dists = np.asarray(distances_knn[j])
    mbias_gan = np.round(np.mean(dists_gan), 2)

    mean_ci_95 = mean_confidence_interval(dists)
    mbias = np.round(np.mean(mean_ci_95[0]), 2)
    cis.append([mean_ci_95[1], mean_ci_95[2]])

    mbiases.append(mbias)
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
  fig2, axes2 = plt.subplots(\
    n_fig_rows, n_fig_cols, figsize=(15,15))

  for j in range(0, dim):
    ax_title = variables[j]
    ax = axes[int(j/n_fig_cols), j % n_fig_cols]
    ax2 = axes2[int(j/n_fig_cols), j % n_fig_cols]
    ax.set_title(ax_title,fontdict={'fontsize':6})

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
    
    imputed_values_knn = np.asarray([imputed_data_x_knn[ii,j]
     for ii in range(0, no) if \
      (not np.isnan(data_x[ii,j]) and \
       data_m[ii,j] == 0)])

    x_range = (
      np.min(imputed_values), 
      np.max(imputed_values)
    )

    qqplot(deleted_values, imputed_values_gan, ax=ax2)

    try:
      kde_kws = {
        'shade': False, 'color': 'r',
        'bw':'scott', 'clip': x_range 
      }
      sns.distplot(imputed_values_gan, \
        kde_kws=kde_kws, hist=False,ax=ax)

      kde_kws = {
        'shade': False, 'color': '#000000',
        'bw':'scott', 'clip': x_range
      }
      sns.distplot(deleted_values, \
        kde_kws=kde_kws, hist=False,ax=ax)

      kde_kws = {
        'shade': False, 'color': 'b',
         'bw':'scott', 'clip': x_range 
      }
      sns.distplot(imputed_values_knn, \
        kde_kws=kde_kws, hist=False,ax=ax)

      kde_kws = {
        'shade': False, 'color': 'y',
         'bw':'scott', 'clip': x_range 
      }
      #sns.distplot(imputed_values_svd, \
      #  kde_kws=kde_kws, hist=False,ax=ax)
    except:
      pass
    ax.set_ylabel('$p(x)$',fontsize=6)
    low_ci, high_ci = cis[j]
    xlab = 'RB = %.2f (%.2f to %.2f)' % \
      (mbiases[j], low_ci, high_ci)
    ax.set_xlabel(xlab,fontsize=6)

  top_title = 'Histogram of original data and imputed data'
  fig.suptitle(top_title, fontsize=8)

  plt.setp(axes, yticks=[], xticks=[])
  fig.tight_layout(rect=[0,0.03,0,1.25])
  fig.subplots_adjust(hspace=1, wspace=0.35)

  top_title = 'Q-Q plot of observed vs. predicted values'
  fig2.suptitle(top_title, fontsize=8)

  plt.setp(axes2, yticks=[], xticks=[])
  fig2.tight_layout(rect=[0,0.03,0,1.25])
  fig2.subplots_adjust(hspace=1, wspace=0.35)
  
  plt.show()

  # Report the RMSE performance
  # rmse = rmse_loss(data_x, imputed_data_x, data_m)
  
  print()
  mrrmse = np.round(np.asarray(rrmses).mean(), 2)
  print('Average RMSE: ' + str(mrrmse) + '%')
  
  return imputed_data_x, rmse

imputed_data, rmse = main()
