import os, csv, sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from gain import gain
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from time_utils import get_datetime_seconds

from constants import SQLITE_DIRECTORY, PLOT_COLORS
from sqlite_utils import sql_fetch_all
from time_utils import get_hours_between_datetimes
from scipy.stats import shapiro

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

BOXCOX_A = 1.5
PLT_ALL = False
include_covid_negative = False
threshold_missingness = 0.1
threshold_discreteness = 0.01

if include_covid_negative:
  inclusion_flag = ""
else:
  inclusion_flag = " AND patient_data.patient_covid_status = 'positive'"

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

query_icu = "SELECT episode_data.patient_site_uid from episode_data INNER JOIN " + \
        " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
        " (episode_data.episode_unit_type = 'intensive_care_unit' OR " + \
        "  episode_data.episode_unit_type = 'high_dependency_unit')  " + \
        inclusion_flag

query_deaths = "SELECT diagnosis_data.patient_site_uid from diagnosis_data INNER JOIN " + \
        " patient_data ON diagnosis_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
        " diagnosis_data.diagnosis_type = 'death' " + \
        inclusion_flag

icu_pt_ids = set([str(x[0]) for x in sql_fetch_all(conn, query_icu)])
death_pt_ids = set([str(x[0]) for x in sql_fetch_all(conn, query_deaths)])

query = "SELECT episode_data.patient_site_uid, episode_start_time from episode_data INNER JOIN " + \
         " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
         " (episode_data.episode_unit_type = 'inpatient_ward' OR " + \
         "  episode_data.episode_unit_type = 'emergency_room') " + \
         inclusion_flag

res = sql_fetch_all(conn, query)

eligible_patients = set()
eligible_episodes = {}

for patient_id, episode_start_time in res:
  patient_id = str(patient_id)
  eligible_patients.add(patient_id)
  episode_start_time = str(episode_start_time)
  if patient_id not in eligible_episodes:
     eligible_episodes[patient_id] = []
  eligible_episodes[patient_id].append(episode_start_time)
  eligible_episodes[patient_id].sort(key = lambda x: \
      get_datetime_seconds(episode_start_time))

query = "SELECT lab_data.patient_site_uid, lab_name, lab_sample_time, lab_result_value, lab_sample_type from lab_data " + \
  " INNER JOIN patient_data ON " + \
  " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
  " (lab_data.lab_sample_type = 'venous_blood' OR " + \
  " lab_data.lab_sample_type = 'arterial_blood' OR " + \
  " lab_data.lab_sample_type = 'unspecified_blood') AND " + \
  " lab_data.lab_result_status = 'resulted' " + \
  inclusion_flag
  
res1 = sql_fetch_all(conn, query)

query = "SELECT observation_data.patient_site_uid, observation_name, observation_time, observation_value from observation_data " + \
  " INNER JOIN patient_data ON " + \
  " observation_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
  " observation_data.observation_value IS NOT NULL " + \
  inclusion_flag
  
res2 = sql_fetch_all(conn, query)

full_lab_data = [[str(value[0]), str(value[1]), str(value[2]), float(value[3]), str(value[4])] for value in res1] + \
                [[str(value[0]), str(value[1]), str(value[2]), float(value[3]), ''] for value in res2]

def limit_to_label(limit):
  left_limit, right_limit, hours_per_period = limit
  n_time_points = (right_limit - left_limit) / hours_per_period
  return 'From %dh to %dh, %d values per variable (1 every %dh)' % \
    (left_limit, right_limit, n_time_points, hours_per_period)


limits = [

  [-24, 36, 12],
  [-24, 80, 16], 
  [-24, 96, 24],

  [0, 36, 12],
  [0, 48, 16], 
  [0, 72, 24],

]

selected_limit = 3

ylabels = []
percentages = []
percentages_total = []

for limit in limits:

  left_limit, right_limit, hours_per_period = limit
  left_offset = int(np.abs(left_limit))
  n_time_points = int((right_limit - left_limit) / hours_per_period)

  lab_bins = {}
  lab_names = []
  patients_with_data = {}
  patient_ids = []
  patient_num = 0
  total_lab_num = 0
  total_entries_num = 0
  percentages.append([])
  ylabels.append(limit_to_label(limit))

  for patient_id, lab_name, lab_sample_time, lab_value, lab_sample_type in full_lab_data:
    if patient_id not in eligible_patients: continue
    if lab_name == 'o2_sat' and lab_sample_type != 'arterial_blood': continue
    if lab_name == 'po2' and lab_sample_type != 'arterial_blood': continue
    if lab_name == 'basophil_count': continue
    if lab_name == 'eosinophil_count': continue
    if lab_name in ['red_blood_cell_count', 'mean_corpuscular_volume', 
      'mean_corpuscular_hemoglobin', 'mean_corpuscular_hemoglobin_concentration']: continue
    
    if patient_id not in lab_bins:
      lab_bins[patient_id] = {}
    if lab_name not in lab_names:
      lab_names.append(lab_name)
    if lab_name not in patients_with_data:
      patients_with_data[lab_name] = set()
    if lab_name not in lab_bins[patient_id]:
      lab_bins[patient_id][lab_name] = [None for x in range(0,n_time_points)]
  
    episode_start_time = eligible_episodes[patient_id][0]
    
    hours_since_admission = get_hours_between_datetimes(episode_start_time, lab_sample_time)
    if hours_since_admission > left_limit and hours_since_admission < right_limit:
      bin_num = int(hours_since_admission / hours_per_period) + int(left_offset / hours_per_period)
      if bin_num >= n_time_points: continue
      if lab_name in lab_bins[patient_id] and \
        bin_num < len(lab_bins[patient_id][lab_name])-1 and \
        lab_bins[patient_id][lab_name][bin_num] is not None:
        # Pick the most abnormal (simplified here to highest)
        if lab_value > lab_bins[patient_id][lab_name][bin_num]: 
          lab_bins[patient_id][lab_name][bin_num] = lab_value
      else: 
        lab_bins[patient_id][lab_name][bin_num] = lab_value
        patients_with_data[lab_name].add(patient_id)
        total_lab_num += 1
      total_entries_num += 1
    patient_ids.append(patient_id)
  
  n_patients = len(np.unique(patient_ids))

  for lab_name in patients_with_data:
    x = len(patients_with_data[lab_name]) / n_patients
    percentages[-1].append(x)
  
  percentages_total.append(\
    total_lab_num / total_entries_num)
    
xlabels = lab_names

patient_ids = np.asarray(np.unique(patient_ids))
lab_names = np.asarray(lab_names)

percentages = np.asarray(percentages)
means = np.asarray([np.mean(percentages[:,x]) \
  for x in range(0, percentages.shape[1])])

for k in range(0, len(limits)):
  
  x, y = [], []

  for i in range(1,9):
    threshold = i * 0.1
    percentages_thresh = percentages[k] < threshold
    
    num_vars_left = np.count_nonzero(percentages_thresh)
    remaining_vars_missingness = percentages_total[k]
    score = remaining_vars_missingness

    x.append(threshold)
    y.append(num_vars_left)
  
  if PLT_ALL: 
    plt.plot(x,y,label=ylabels[k], \
      marker='o', color=PLOT_COLORS[k])

if PLT_ALL: 
  plt.title("Number of features included according to different sampling strategies and maximal missingness threshold")
  plt.ylabel("Number of features that would be included given maximum missingness threshold")
  plt.xlabel("Maximum fraction of missing data allowed for each feature (as % of patients with no data)")
  plt.xlim(0.1, 0.9)
  plt.legend()
  plt.show()

left_limit, right_limit, hours_per_period = limits[selected_limit]
means_thresh = percentages[selected_limit,:] > threshold_missingness
selected_variables = lab_names[means_thresh].tolist()

n_selected_variables = len(selected_variables)
print('Total %d variables selected' % n_selected_variables)

print(selected_variables)

if PLT_ALL: 

  fig, ax = plt.subplots()
  xy_data = percentages[:,means_thresh]
  im = ax.matshow(xy_data)
  xlabels = selected_variables
  ax.set_xticks(np.arange(len(xlabels)))
  ax.set_yticks(np.arange(len(ylabels)))
  ax.set_xticklabels(xlabels, fontsize=6)
  ax.set_yticklabels(ylabels, fontsize=6)
  ax.tick_params(axis='x', bottom=True, \
    labelbottom=True, top=False, labeltop=False)
  plt.setp(ax.get_xticklabels(), rotation=45, \
    ha='right', rotation_mode='anchor')
  fig.colorbar(im, orientation='horizontal')

  ax.set_title("Percentage patients with at least one value for each variable")
  fig.tight_layout()
  plt.show()

pct_missingness_threshold = 0.5
pcts_completely_missing = []
selected_patient_ids = []

for patient_id in patient_ids:
  
  n_variables_total = 0 
  n_variables_completely_missing = 0

  for variable_num in range(0, len(selected_variables)):
    variable_name = selected_variables[variable_num]
    any_value_present = False
    if variable_name not in lab_bins[patient_id]:
      n_variables_completely_missing += 1
    elif all(v is None for v in lab_bins[patient_id][variable_name]):
      n_variables_completely_missing += 1
    n_variables_total += 1
  
  pct_variables_completely_missing = \
    n_variables_completely_missing / n_variables_total
  
  if pct_variables_completely_missing < pct_missingness_threshold:
    selected_patient_ids.append(patient_id)
    pcts_completely_missing.append(pct_variables_completely_missing)

num_patients_original = len(patient_ids)
print('Total %d patients initially' % num_patients_original)

num_patients_left = len(selected_patient_ids)
print('Total %d patients remaining' % num_patients_left)

if PLT_ALL: 

  n, bins, patches = plt.hist(x=pcts_completely_missing, 
    bins='auto', color=PLOT_COLORS[0], rwidth=0.9)
  plt.grid(axis='y', alpha=0.4)
  plt.xlabel('Fraction of selected variables completely missing')
  plt.ylabel('Number of patients')
  plt.title(('Fraction of selected variables (n=%d) completely ' + \
    'missing in eligible patients') % len(selected_variables))

  max_freq = n.max()
  max_val = np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10

  plt.ylim(ymax=max_val + 5)
  plt.show()

pct_missing_timepoints = []

for variable_name in selected_variables:
  pct_missing_timepoints_for_variable = \
    np.zeros(n_time_points)
  for patient_id in selected_patient_ids:
    if variable_name not in lab_bins[patient_id]: 
      pct_missing_timepoints_for_variable += 1
    else:
      variable_values = lab_bins[patient_id][variable_name]
      for i in range(0, len(variable_values)):
        if variable_values[i] is None:
          pct_missing_timepoints_for_variable[i] += 1

  pct_missing_timepoints.append(\
    pct_missing_timepoints_for_variable / 
    len(selected_patient_ids))
  patient_num += 1

if PLT_ALL: 
 fig, ax = plt.subplots()
 im = ax.matshow(pct_missing_timepoints)

time_points_labels = ['t=%dh' % \
  (i * hours_per_period + left_limit) \
  for i in range(0, n_time_points)]

if PLT_ALL: 

  ax.set_xticks(np.arange(len(time_points_labels)))
  ax.set_yticks(np.arange(len(selected_variables)))

  ax.set_xticklabels(time_points_labels, fontsize=8)
  ax.set_yticklabels(selected_variables, fontsize=6)

  ax.tick_params(axis='x', bottom=True, \
    labelbottom=True, top=False, labeltop=False)

  plt.setp(ax.get_xticklabels(), rotation=45, \
    ha='right', rotation_mode='anchor')

  fig.colorbar(im, orientation='vertical')

  ax.set_title("Percentage of patients with data for each variable at each time point")
  fig.tight_layout()
  plt.show()

values_for_variables = []
discrete_like_variables = []

for variable_name in selected_variables:
  values_for_variable = []
  for patient_id in selected_patient_ids:
    if variable_name in lab_bins[patient_id]:
      v = lab_bins[patient_id][variable_name]
      values_for_variable.extend(v)
    else:
      values_for_variable.extend([None for i \
        in range(0,n_time_points)])
  x_not_none = [x for x \
    in values_for_variable if x is not None]
  n_unique = len(set(x_not_none))
  pct_unique = n_unique / len(x_not_none)
  if pct_unique < threshold_discreteness:
    discrete_like_variables.append(variable_name)
  else: 
    values_for_variables.append(values_for_variable)

for variable_name in discrete_like_variables:
  selected_variables.remove(variable_name)
  print('Removed variable %s (discrete-like)' %\
    variable_name)

n_fig_rows = 7
n_fig_cols = 7

n_fig_total = n_fig_rows * n_fig_cols

if len(values_for_variables) > n_fig_total:
  print('Warning: not all variables plotted')

import scipy.stats as st
from scipy.stats import kstest

def find_best_dist(variable_name, values_for_variable):
  
  plot_distribution = False

  distribution_names = [
    'norm','halfnorm', 'foldnorm', 'truncnorm', 'expon',
    'lognorm', 'powerlognorm', 'powernorm', 'exponnorm'
    #'laplace','t','alpha', 'cauchy',
    #'cosine', 'levy', 'invgauss',  'genextreme', 'chi2'
  ]

  distributions = [
    st.norm,st.halfnorm,st.foldnorm,st.truncnorm, st.expon,
    st.lognorm, st.powerlognorm, st.powernorm, st.exponnorm
    #st.laplace,st.t,st.alpha, st.cauchy,
    #st.cosine, st.levy, st.invgauss, st.genextreme, st.chi2
  ]

  discrete_distribution_names = [
    'poisson', 'logser', 'randint'
  ]

  #discrete_distributions = [
  #  st.poisson, st.logser, st.randint
  #]

  mles = []
  all_params = []
  dist_num = 0

  for distribution in distributions:

    params = distribution.fit(values_for_variable)
    all_params.append(params)

    mle = distribution.nnlf(params,values_for_variable)

    dist_name = distribution_names[dist_num]
    mle = kstest(values_for_variable, dist_name, args=params)[1]

    mles.append(mle)

    #if plot_statistics:
      #plt.plot(dist_num, mle, marker='o', color='k')
      #plt.plot(dist_num, p_ks, marker='o', color='r')

    if plot_distribution:
      fig, ax = plt.subplots()
    
      x_range = [0, np.max(values_for_variable)]
      x = np.linspace(x_range[0], x_range[1], 1000)
      y = distribution.pdf(x,*params[:-2],
        loc=params[-2], scale=params[-1])
      plt.title(variable_name + '-' + dist_name + ': ' + str(mle))
      plt.hist(values_for_variable,alpha=0.5, density=True)
      plt.plot(x, y, c='k', lw=2)
      plt.gca().set_ylim(0, np.max(y))
      plt.show()

    dist_num += 1
  
  #if plot_statistics:
  #  plt.xticks([x for x in range(0, dist_num)], labels=distribution_names)
  #  plt.show()

  results = [(distribution.name, mle) for \
    distribution, mle in zip(distributions,mles)]

  best_fits = sorted(zip(distributions,mles),
    key=lambda d: -d[1])
  
  best_fits_params = sorted(zip(all_params,mles),
    key=lambda d: -d[1])

  best_fits_names = sorted(zip(distribution_names,mles),
    key=lambda d: -d[1])

  return [best_fits[0][0], best_fits_params[0][0], \
           best_fits_names[0][0], max(mles)]

if PLT_ALL: 
  fig, axes = plt.subplots(\
    n_fig_rows, n_fig_cols, figsize=(15,15))
  top_title = 'Histogram of data for selected variables, ' + \
    'kernel density estimation, and best fit distribution\n\n'
  #top_title = "$\mathbf{" + top_title.replace(' ', '\\') + "}$\n\n"
  sub_title = '$N_T$=%s, $N_U$=%s, $N_O$=%s\n$M$=%s, $kurt$=%s, $skew$=%s' % \
    ('number of observations', 'number of unique values',
      'number of right-tail outliers not plotted',
      'percentage of missing values', 'distribution kurtosis', 
      'distribution skewness')
  fig.suptitle(top_title + sub_title, fontsize=8)

remove_large_outliers = True
export_values_for_variables = []
all_values_for_variables = []
positive_values_for_variables = []
scaling_parameters = []

from scipy.stats import boxcox_normmax

for k in range(0, len(selected_variables)):
  variable_name = selected_variables[k]
  values_for_variable = values_for_variables[k]
  export_values_for_variables.append(values_for_variable)
  
  all_values_for_variable = np.asarray(values_for_variable)

  nn_values_for_variable = np.asarray([v for \
    v in values_for_variable if v is not None])
  n_outliers_removed = 0

  #if remove_large_outliers:
  #  percentile_99 = np.percentile(nn_values_for_variable, 97)
  #  outlier_indices = nn_values_for_variable > percentile_99
  #  n_outliers_removed = np.count_nonzero(outlier_indices)
  #  nn_values_for_variable = nn_values_for_variable[outlier_indices == False]
  
  nn_values_for_variable = np.asarray(nn_values_for_variable)
  
  alpha = np.min(nn_values_for_variable)
  nn_values_for_variable -= alpha

  lmbda = boxcox_normmax(nn_values_for_variable+BOXCOX_A, method='mle')
  nn_values_for_variable = boxcox(nn_values_for_variable+BOXCOX_A,lmbda=lmbda)
  
  ind = (all_values_for_variable != None)
  
  all_values_for_variable[ind] = nn_values_for_variable
  #beta = np.std(nn_values_for_variable)
  #nn_values_for_variable /= beta
  #delta = np.mean(nn_values_for_variable)
  #nn_values_for_variable -= delta
  all_values_for_variables.append(all_values_for_variable)
  positive_values_for_variables.append(nn_values_for_variable)
  scaling_parameters.append([lmbda, alpha, 1, 0])

  n_unique_values = len(np.unique(nn_values_for_variable))
  n_not_missing = len(nn_values_for_variable)
  n_total_samples = len(values_for_variable)
  pct_missing = (1-n_not_missing / n_total_samples) * 100
  p_is_norm = shapiro(nn_values_for_variable)[1]
  dist_kurt = st.kurtosis(nn_values_for_variable)
  dist_skew = st.skew(nn_values_for_variable)
  
  best_dist, best_dist_params, best_dist_name, mle = \
    find_best_dist(variable_name, nn_values_for_variable)
  
  if PLT_ALL: 

    if len(variable_name) > 15: 
      short_var_name = variable_name[0:15] + '...'
    else: short_var_name = variable_name

    ax_title = '%s ($N_T$=%d, M=%.2f%%)' % \
      (short_var_name, n_not_missing, pct_missing)

    ax = axes[int(k/n_fig_cols), k % n_fig_cols]
    ax.set_title(ax_title,fontdict={'fontsize':6})

    x_range = (
      np.min(nn_values_for_variable), 
      np.max(nn_values_for_variable)
    )

    x = np.linspace(x_range[0], x_range[1], 1000)
    y = best_dist.pdf(x,*best_dist_params[:-2],
      loc=best_dist_params[-2], scale=best_dist_params[-1])
    ax.plot(x, y, c=PLOT_COLORS[5], lw=2)

    hist_kws = {'color': PLOT_COLORS[0], 'alpha':0.4, 'range': x_range}
    kde_kws = {'shade': False, 'color': '#000000','bw':'scott', 'clip': x_range}
    sns.distplot(nn_values_for_variable, \
      kde_kws=kde_kws, hist_kws=hist_kws,ax=ax)
    ax.set_ylabel('$p(x)$',fontsize=6)
    ax.set_xlabel(
      ('$N_U$=%d, fit = %s, SW p = %.4f') % \
      (n_unique_values, best_dist_name, p_is_norm),
      fontsize=6
    )
    ax.text(0.75, 0.40, 
      '$N_O$=%d\nkurt=%.1f\nskew=%.1f' % \
      (n_outliers_removed, dist_kurt, dist_skew),
      transform=ax.transAxes,
      fontdict={'fontsize': 5}
    )
   
  #min_val = np.min([i for i in values_for_variables[k] if i is not None])
  #vs = positive_values_for_variables[k]
  #bc_values_for_variable, lmbda = boxcox(np.asarray(vs)+1)

  #sns.distplot(bc_values_for_variable, hist=False, 
  #  kde_kws={ 'shade': False, 'lw': 1, \
  #  'color': PLOT_COLORS[4],'bw':'scott' },ax=ax)

  pct_missing = float(len(nn_values_for_variable)) \
   / float(len(values_for_variable)) * 100
  min_value = np.min(nn_values_for_variable)
  max_value = np.max(nn_values_for_variable)

  #print('Feature %s, pct. missing: %.2f, min: %.2f, max: %.2f' % 
  #  (variable_name, pct_missing, min_value, max_value))
import pickle
export_values_for_variables = np.asarray(export_values_for_variables)
pickle.dump(export_values_for_variables, open('missing_data.sav', 'wb'))

if PLT_ALL: 

  plt.setp(axes, yticks=[], xticks=[])
  plt.tight_layout(rect=[0,0.03,0,1.25])
  plt.subplots_adjust(hspace=1, wspace=0.35)
  plt.show()
 
X = np.asarray(all_values_for_variables).astype(np.float).transpose()

gain_parameters = { 'batch_size': 128,
                    'hint_rate': 0.9,
                    'alpha': 1000,
                    'iterations': 1000}

#imputer = KNNImputer(n_neighbors=5)
#imputed_variable_values = imputer.fit_transform(X).transpose()
imputed_variable_values = gain(X, gain_parameters).transpose()

df = pd.DataFrame(imputed_variable_values)
patients_labels = []
patients_features = []

PLT_ALL = True

for i in range(0, len(selected_patient_ids)):

  patient_id = selected_patient_ids[i]
  patient_features = []

  for k in range(0, len(selected_variables)):
    variable_name = selected_variables[k]
    imputed_values_for_variable = df.values[k,:]
    original_values_for_variable = values_for_variables[k]
   
    j = i*n_time_points
    orig_pt_var = original_values_for_variable[j:j+n_time_points]
    imput_pt_var = imputed_values_for_variable[j:j+n_time_points]
    
    original_values = orig_pt_var[1:4]
    generated_values = imput_pt_var[1:4]
    
    lmbda, alpha, beta, delta = scaling_parameters[k]
    #generated_values += delta
    #generated_values *= beta
    bc_generated_values = inv_boxcox(
      generated_values, lmbda) - BOXCOX_A
    al_generated_values = bc_generated_values + alpha
    generated_values = al_generated_values

    for kk, x in enumerate(generated_values):
      if x is not None:
        if np.isnan(x):
          generated_values[kk] = generated_values[kk-1]
          print('Found NAN for variable 2 ' + str(kk))
          print(orig_pt_var[1:4])
          print(imput_pt_var[1:4])
          print(bc_generated_values)
          print(al_generated_values)
          print(generated_values)
          
    patient_features.append(generated_values)

  patients_features.append(np.asarray(patient_features))
  patients_labels.append(patient_id in death_pt_ids or patient_id in icu_pt_ids)

if PLT_ALL: fig, axes = plt.subplots(\
  n_fig_rows, n_fig_cols, figsize=(15,15))

for k in range(0, len(selected_variables)):

  observed_variable_values = []
  for i in range(0, len(selected_patient_ids)):
    observed_variable_values.extend(patients_features[i][k])
   
  observed_variable_values = np.asarray(observed_variable_values)

  if PLT_ALL: 
    ax = axes[int(k/n_fig_cols), k % n_fig_cols]
    ax_title = selected_variables[k]
    ax.set_title(ax_title,fontdict={"fontsize":5})
    sns.distplot(observed_variable_values, kde_kws={ \
      'shade': False, 'color': '#000000','bw':'scott' \
    }, hist_kws={'color': PLOT_COLORS[0], 'alpha':0.4},ax=ax)
  
    ax.set_ylabel('$p(x)$',fontsize=6)

if PLT_ALL:
  plt.tight_layout(rect=[0,0.03,0,1.25])
  plt.subplots_adjust(hspace=1, wspace=0.35)
  plt.suptitle('Distribution of imputed values', fontsize=8)
  plt.show()

X = np.asarray([x.flatten() for x in patients_features])
y = np.asarray(patients_labels).astype(np.float)

kfold = KFold(n_splits=5)

#X_train, X_test, y_train, y_test = train_test_split( \
#  classifier_input, classifier_labels, test_size=1/3)

print('Classifier input array shape: %s' % str(X.shape))
print('Classifier label array shape: %s' % str(y.shape))

clf = RandomForestClassifier(max_depth=10, random_state=0)

plt.figure()
fold = 0
tprs = []
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in kfold.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  y_score = clf.fit(X_train, y_train).predict_proba(X_test)

  fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
  tprs.append(np.interp(mean_fpr, fpr, tpr))
  roc_auc = auc(fpr, tpr)

  plt.plot(fpr, tpr, lw=1, color='gray', linestyle='--', \
    label='Fold %i (AUC = %0.2f)' % (fold, roc_auc))

  fold += 1

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, lw=2, color='green', \
  label='Average %i (AUC = %0.2f)' % (fold, mean_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Prediction of death')
plt.legend(loc="lower right")
plt.show()