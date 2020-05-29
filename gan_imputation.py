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
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from scipy.stats import boxcox, t, sem
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
  outlier_indices = mse > 0.05
  
  #plt.title(variable_name); plt.plot(mse); plt.show()
  return outlier_indices

def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, color=None, **kwargs):
    
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
    ax.plot(x, x, lw=1,color=color)

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
    
# 5000
def main (alpha=1000, batch_size=128, hint_rate=0.5, 
  iterations=5500, miss_rate=0.3):
  
  gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'alpha': alpha,
                     'iterations': iterations}
  
  # Load data and introduce missingness
  #file_name = 'data/spam.csv'
  #data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  
  remove_outliers = False
  n_time_points = 3
  
  data_x = pickle.load(open('./missing_data.sav', 'rb'))
  data_x = data_x.transpose().astype(np.float)
  print(data_x.shape)
  # if remove_outliers:
  #  data_x = pickle.load(open('./missing_data.sav', 'rb'))
  #  data_x = data_x.transpose().astype(np.float)
  # else:
  #  data_x = pickle.load(open('./denoised_missing_data.sav', 'rb')) 

  signed_variables = ['base_excess']
  no, dim = data_x.shape
  
  data_x_encoded = np.copy(data_x)
  miss_data_x = np.copy(data_x)
  miss_data_x_enc = np.copy(data_x_encoded)
  
  scalers = []
  
  for i in range(0, dim):
      variable, var_x = variables[i], data_x[:,i]
      encoder_model = encoders[i]
      # Exclude outliers based on error
      nn_indices = ~np.isnan(data_x_encoded[:,i])
      nn_values = data_x[:,i][nn_indices]
      
      scaler = MinMaxScaler()
      var_x_scaled = scaler.fit_transform(var_x.reshape((-1,1)))
      enc_x_scaled = encoder_model.predict(var_x_scaled)
      enc_x_unscaled = scaler.inverse_transform(enc_x_scaled)
      data_x_encoded[:,i] = enc_x_unscaled.flatten()
      
      # mse = np.mean(np.power(var_x.reshape((-1,1)) - enc_x_unscaled, 2),axis=1)
      # 
      # x = np.ma.array(mse, mask=np.isnan(mse))
      # y = np.ma.array(var_x, mask=np.isnan(var_x))
      # outlier_indices = (x / np.max(y)) > 2
      # 
      # outlier_values = var_x[outlier_indices]
      # 
      # print('... %d outlier(s) excluded' % \
      #   len(outlier_values), outlier_values)
      # 
      # miss_data_x[outlier_indices == True,i] = np.nan
      # miss_data_x_enc[outlier_indices == True,i] = np.nan
      
      scalers.append(scaler)
      #print(var_x, '----', enc_x_scaled, '----', enc_x_unscaled.flatten())
      print('Loaded model for %s...' % variable)
  
  no_total = no * dim
  no_nan = np.count_nonzero(np.isnan(data_x.flatten()) == True)
  no_not_nan = no_total - no_nan
  print('Input shape', no, 'x', dim)
  print('NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))

  n_patients = int(no/n_time_points)

  if len(variables) != dim:
    print(len(variables), dim)
    print('Incompatible dimensions.')
    exit()
    
  #transformer = PowerTransformer(method='box-cox')
  #transformer.fit(miss_data_x)
  
  #data_x = transformer.transform(data_x)
  #miss_data_x = transformer.transform(miss_data_x)
  #miss_data_x_enc = transformer.transform(data_x_encoded)
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)

  miss_data_x[data_m == 0] = np.nan
  miss_data_x_enc[data_m == 0] = np.nan

  no_nan = np.count_nonzero(np.isnan(miss_data_x.flatten()) == True)
  no_not_nan = no_total - no_nan

  print('After removal, NAN values:', no_nan, '/', no_total, \
    '%2.f%%' % (no_nan / no_total * 100))
  
  real_miss_rate = (no_nan / no_total * 100)
  
  imputed_data_x_gan = gain(
    miss_data_x_enc, gain_parameters)
  
  # n_gans = 3
  # idxg_combined = []
  # 
  # for  n_gan in range(0, n_gans):
  #   np.random.seed(n_gan + 1)
  #   idxg_combined.append(gain(miss_data_x_enc, gain_parameters))
  # 
  # idxg_combined = np.concatenate(idxg_combined)
  #   
  # idxg_combined_final = gain(
  #   miss_data_x_enc, gain_parameters)
  # 
  # for j in range(0, dim):
  #   idxg_combined_tmp = np.copy(idxg_combined)
  #   
  #   for i in range(0, n_patients * n_time_points):
  #     if np.isnan(miss_data_x[i,j]) and data_m[i,j] != 0:
  #       idxg_combined_tmp[i,j] = np.nan
  # 
  #   imputer = IterativeImputer() # KNNImputer(n_neighbors=5)
  #   idxg_knn = imputer.fit_transform(idxg_combined_tmp)
  #   idxg_combined_final[:,j] = idxg_knn[0:n_patients*n_time_points,j]
  #   print('Done KNN imputation #%d' % j)
  # 
  # imputed_data_x_gan = idxg_combined_final

  imputer = KNNImputer(n_neighbors=5)
  imputed_data_x_knn = imputer.fit_transform(miss_data_x_enc)
  
  imputer = IterativeImputer()
  imputed_data_x_mice = imputer.fit_transform(miss_data_x_enc)
  
  # data_x = transformer.inverse_transform(data_x)-1
  # miss_data_x = transformer.inverse_transform(miss_data_x)-1
  # imputed_data_x_gan = transformer.inverse_transform(imputed_data_x_gan)-1
  # imputed_data_x_knn = transformer.inverse_transform(imputed_data_x_knn)-1
  # imputed_data_x_mice = transformer.inverse_transform(imputed_data_x_mice)-1
  
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
      
      print(original_tuple, corrupted_tuple, imputed_tuple_gan, imputed_tuple_knn)
      for k in range(0, n_time_points):
        a, b, c, d = original_tuple[k], imputed_tuple_gan[k], imputed_tuple_knn[k], imputed_tuple_mice[k]
        if np.isnan(a): continue
        distances_gan[j,i*k] = (b - a)
        distances_knn[j,i*k] = c - a
        distances_mice[j,i*k] = d - a
  
  # Compute distance statistics
  rrmses_gan, mean_biases, median_biases, bias_cis = [], [], [], []
  rrmses_knn, mean_biases_knn, median_biases_knn, bias_cis_knn = [], [], [], []

  for j in range(0, dim):
    
    # Stats for original data
    dim_mean = np.mean([x for x in data_x[:,j] if not np.isnan(x)])
    
    # Stats for GAN
    dists_gan = np.asarray(distances_gan[j])
    mean_bias = np.round(np.mean(dists_gan), 4)
    median_bias = np.round(np.median(dists_gan), 4)
    mean_ci_95 = mean_confidence_interval(dists_gan)
    rmse = np.sqrt(np.mean(dists_gan**2))
    rrmse = np.round(rmse / dim_mean * 100, 2)
    
    bias_cis.append([mean_ci_95[1], mean_ci_95[2]])
    mean_biases.append(mean_bias)
    median_biases.append(median_bias)
    rrmses_gan.append(rrmse)
    
    # Stats for KNN
    dists_knn = np.asarray(distances_knn[j])
    rmse_knn = np.sqrt(np.mean(dists_knn**2))
    rrmses_knn = np.round(rmse_knn / dim_mean * 100, 2)
    
    # Stats for MICE
    dists_mice = np.asarray(distances_mice[j])
    rmse_mice = np.sqrt(np.mean(dists_mice**2))
    rrmses_mice = np.round(rmse_mice / dim_mean * 100, 2)
    
    print(variables[j], ' - rrmse: ', rrmse, 
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

  for j in range(0, dim):
    
    ax_title = variables[j]
    ax = axes[int(j/n_fig_cols), j % n_fig_cols]
    ax2 = axes2[int(j/n_fig_cols), j % n_fig_cols]
    ax.set_title(ax_title,fontdict={'fontsize':6})

    input_arrays = [data_x, imputed_data_x_gan, imputed_data_x_knn, imputed_data_x_mice]
    
    output_arrays = [
      np.asarray([input_arr[ii,j] for ii in range(0, no) if \
        (not np.isnan(input_arr[ii,j]) and \
        data_m[ii,j] == 0)]) for input_arr in input_arrays
    ]
    
    deleted_values, imputed_values_gan, imputed_values_knn, imputed_values_mice = output_arrays
    
    # Make KDE
    low_ci, high_ci = bias_cis[j]
    xlabel = 'mean bias = %.2f (95%% CI, %.2f to %.2f)' % \
      (mean_biases[j], low_ci, high_ci)
      
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel('$p(x)$',fontsize=6)
    
    range_arrays = np.concatenate([deleted_values, imputed_values_gan])
    
    x_range = (np.min(range_arrays), 
      np.min([
        np.mean(range_arrays) + 3 * np.std(range_arrays), 
        np.max(range_arrays)
      ])
    )
    
    kde_kws = { 'shade': False, 'bw':'scott', 'clip': x_range }
    
    sns.distplot(imputed_values_gan, hist=False,
      kde_kws={**{ 'color': 'r'}, **kde_kws}, ax=ax)
    
    sns.distplot(imputed_values_knn, hist=False,
      kde_kws={**{ 'color': 'b', 'alpha': 0.5 }, **kde_kws},ax=ax)

    sns.distplot(imputed_values_mice, hist=False,
      kde_kws={**{ 'color': 'g', 'alpha': 0.5 }, **kde_kws},ax=ax)

    sns.distplot(deleted_values, hist=False,
      kde_kws={**{ 'color': '#000000'}, **kde_kws},ax=ax)

    # Make QQ plot
    qqplot(deleted_values, imputed_values_gan, ax=ax2, color='r')
    qqplot(deleted_values, imputed_values_knn, ax=ax2, color='b')
    qqplot(deleted_values, imputed_values_mice, ax=ax2, color='g')
    
  top_title = 'KDE plot of original data (black) and data imputed using GAN (red) and KNN (blue)'
  fig.suptitle(top_title, fontsize=8)
  fig.legend(labels=['GAN', 'KNN', 'MICE', 'Observed'])

  fig.tight_layout(rect=[0,0.03,0,1.25])
  fig.subplots_adjust(hspace=1, wspace=0.35)

  top_title = 'Q-Q plot of observed vs. predicted values'
  fig2.suptitle(top_title, fontsize=8)

  fig2.tight_layout(rect=[0,0.03,0,1.25])
  fig2.subplots_adjust(hspace=1, wspace=0.35)
  
  plt.show()

  print()
  mrrmse_gan = np.round(np.asarray(rrmses_gan).mean(), 2)
  print('Average RMSE (GAN): ', mrrmse_gan, '%')

  print()
  mrrmse_knn = np.round(np.asarray(rrmses_knn).mean(), 2)
  print('Average RMSE (KNN): ', mrrmse_knn, '%')

  print()
  mrrmse_mice = np.round(np.asarray(rrmses_mice).mean(), 2)
  print('Average RMSE (MICE): ', mrrmse_mice, '%')
  
  return real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice

errors = []
for k in np.linspace(0.1,0.9, 9):
  print('----------')
  real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice = main(miss_rate = 0.2)
  errors.append([real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice])
  print(real_miss_rate, mrrmse_gan, mrrmse_knn, mrrmse_mice)
  
print(errors)