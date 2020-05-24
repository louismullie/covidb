import os, csv, sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from constants import SQLITE_DIRECTORY, PLOT_COLORS
from sqlite_utils import sql_fetch_all
from time_utils import get_hours_between_datetimes
from scipy.stats import shapiro

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
from sklearn.impute import KNNImputer

BOXCOX_A = 1.5
PLT_ALL = False

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

query_icu = "SELECT episode_data.patient_site_uid from episode_data INNER JOIN " + \
        " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
        " (episode_data.episode_unit_type = 'intensive_care_unit' OR " + \
        "  episode_data.episode_unit_type = 'high_dependency_unit') AND " + \
        " patient_data.patient_covid_status = 'positive'"

query_deaths = "SELECT diagnosis_data.patient_site_uid from diagnosis_data INNER JOIN " + \
        " patient_data ON diagnosis_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
        " diagnosis_data.diagnosis_type = 'death' AND " + \
        " patient_data.patient_covid_status = 'positive'"

icu_pt_ids = set([str(x[0]) for x in sql_fetch_all(conn, query_icu)])
death_pt_ids = set([str(x[0]) for x in sql_fetch_all(conn, query_deaths)])

query = "SELECT episode_data.patient_site_uid, episode_start_time from episode_data INNER JOIN " + \
         " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
         " (episode_data.episode_unit_type = 'inpatient_ward' OR " + \
         " episode_data.episode_unit_type = 'intensive_care_unit') AND " + \
         " patient_data.patient_covid_status = 'positive'"

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

query = "SELECT lab_data.patient_site_uid, lab_name, lab_sample_time, lab_result_value, lab_sample_type from lab_data " + \
  " INNER JOIN patient_data ON " + \
  " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
  " (lab_data.lab_sample_type = 'venous_blood' OR " + \
  " lab_data.lab_sample_type = 'arterial_blood' OR " + \
  " lab_data.lab_sample_type = 'unspecified_blood') AND " + \
  " lab_data.lab_result_status = 'resulted' AND " + \
  " patient_data.patient_covid_status = 'positive'"
  
res1 = sql_fetch_all(conn, query)

query = "SELECT observation_data.patient_site_uid, observation_name, observation_time, observation_value from observation_data " + \
  " INNER JOIN patient_data ON " + \
  " observation_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
  " observation_data.observation_value IS NOT NULL AND " + \
  " patient_data.patient_covid_status = 'positive'"
  
res2 = sql_fetch_all(conn, query)

full_lab_data = [[str(value[0]), str(value[1]), str(value[2]), float(value[3]), str(value[4])] for value in res1] # + \
           #[[str(value[0]), str(value[1]), str(value[2]), float(value[3])] for value in res2]

def limit_to_label(limit):
  left_limit, right_limit, hours_per_period = limit
  n_time_points = (right_limit - left_limit) / hours_per_period
  return 'From %dh to %dh, %d values per variable (1 every %dh)' % \
    (left_limit, right_limit, n_time_points, hours_per_period)

limits = [
  [0, 36, 12], 
  [-12, 36, 12], 
  [-24, 36, 12], 
  [0, 48, 16], 
  [-16, 48, 16], 
  [-32, 48, 16], 
  [0, 72, 24], 
  [-24, 72, 24], 
  [-48, 72, 24]
]

ylabels = []
percentages = []

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
    
    lab_names.append(lab_name)

    if patient_id not in lab_bins:
      lab_bins[patient_id] = {}
    if lab_name not in patients_with_data:
      patients_with_data[lab_name] = set()
    if lab_name not in lab_bins[patient_id]:
      lab_bins[patient_id][lab_name] = [None for x in range(0,n_time_points)]
  
    episode_start_time = eligible_episodes[patient_id][0]
    
    hours_since_admission = get_hours_between_datetimes(episode_start_time, lab_sample_time)
    if hours_since_admission > left_limit and hours_since_admission < right_limit:
      bin_num = int(hours_since_admission / hours_per_period) + int(left_offset / hours_per_period)
      if lab_bins[patient_id][lab_name][bin_num] is not None:
        if lab_value > lab_bins[patient_id][lab_name][bin_num]: 
          lab_bins[patient_id][lab_name][bin_num] = lab_value
      else:
        lab_bins[patient_id][lab_name][bin_num] = lab_value
        patients_with_data[lab_name].add(patient_id)
        total_lab_num += 1
      total_entries_num += 1
    patient_ids.append(patient_id)

  n_patients = len(np.unique(patient_ids))
  print(total_lab_num / total_entries_num)

  lab_names = []

  for lab_name in patients_with_data:
    x = len(patients_with_data[lab_name]) / n_patients
    percentages[-1].append(x)
    lab_names.append(lab_name)

  xlabels = lab_names

patient_ids = np.asarray(np.unique(patient_ids))
lab_names = np.asarray(lab_names)

percentages = np.asarray(percentages)
means = np.asarray([np.mean(percentages[:,x]) \
  for x in range(0, percentages.shape[1])])

for k in range(0,percentages.shape[0]):
  points = []
  for i in range(1, 9):
    threshold = i * 0.1
    percentages_thresh = percentages[k]
    percentages_thresh = percentages_thresh > threshold
    num_vars_left = np.count_nonzero(percentages_thresh)
    points.append([threshold, num_vars_left])
  points = np.asarray(points)
  if PLT_ALL: plt.plot(points[:,0],points[:,1], label=ylabels[k], \
    marker='o', color=PLOT_COLORS[k])

if PLT_ALL: 
  plt.title("Missing data for different sampling strategies")
  plt.ylabel("Number of features exceeding threshold")
  plt.xlabel("Fraction of values that are not missing")

  plt.legend()
  plt.show()

threshold_missingness = 0.25
threshold_discreteness = 0.075
means_thresh = means > threshold_missingness

selected_variables = lab_names[means_thresh].tolist()
xy_data = percentages[:,means_thresh]

n_selected_variables = len(selected_variables)
print('Total %d variables selected' % n_selected_variables)

print(selected_variables)

if PLT_ALL: 

  fig, ax = plt.subplots()
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


pct_missingness_threshold = 1.0
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

num_patients_left = len(selected_patient_ids)
print('Total %d patients remaining' % num_patients_left)

if PLT_ALL: 

  n, bins, patches = plt.hist(x=pcts_completely_missing, 
    bins='auto', color=PLOT_COLORS[0], rwidth=0.9)
  plt.grid(axis='y', alpha=0.75)
  plt.xlabel('Fraction of selected variables completely missing')
  plt.ylabel('Number of patients')
  plt.title(('Fraction of selected variables (n=%d) completely ' + \
    'missing in eligible patients') % len(selected_variables))

  max_freq = n.max()
  max_val = np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10

  plt.ylim(ymax=max_val + 5)
  plt.show()

pct_missing_timepoints = []
n_time_points = 5

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
n_fig_cols = 6

n_fig_total = n_fig_rows * n_fig_cols

if len(values_for_variables) > n_fig_total:
  print('Warning: not all variables plotted')

PLT_ALL = False

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
positive_values_for_variables = []
scaling_parameters = []

from scipy.stats import boxcox_normmax

for k in range(0, len(selected_variables)):
  variable_name = selected_variables[k]
  values_for_variable = values_for_variables[k]
  
  nn_values_for_variable = np.asarray([v for \
    v in values_for_variable if v is not None])
    
  n_outliers_removed = 0

  if remove_large_outliers:
    percentile_99 = np.percentile(nn_values_for_variable, 97)
    outlier_indices = nn_values_for_variable > percentile_99
    n_outliers_removed = np.count_nonzero(outlier_indices)
    nn_values_for_variable = nn_values_for_variable[outlier_indices == False]
  
  nn_values_for_variable = np.asarray(nn_values_for_variable)
  
  alpha = np.min(nn_values_for_variable)
  nn_values_for_variable -= alpha

  lmbda = boxcox_normmax(nn_values_for_variable+BOXCOX_A, method='pearsonr')
  nn_values_for_variable = boxcox(nn_values_for_variable+BOXCOX_A,lmbda=lmbda)
  
  #beta = np.std(nn_values_for_variable)
  #nn_values_for_variable /= beta
  #delta = np.mean(nn_values_for_variable)
  #nn_values_for_variable -= delta

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

if PLT_ALL: 
  plt.setp(axes, yticks=[], xticks=[])
  plt.tight_layout(rect=[0,0.03,0,1.25])
  plt.subplots_adjust(hspace=1, wspace=0.35)
  plt.show()
 
def freedman_diaconis_bins(a): 

  a = np.asarray(a) 
  if len(a) < 2:  return 1 

  h = 2 * st.iqr(a) / (len(a) ** (1 / 3)) 
  
  if h == 0: 
    return int(np.sqrt(a.size)) 
  else: 
    return int(np.ceil((a.max() - a.min()) / h)) 

labels = [None for k in range(0, n_fig_total)]

def fit_gmm(variable_name,values_for_variable,ax1, ax2,show_plot=True):
  
  random_state = np.random.RandomState(seed=1)
  X = values_for_variable.reshape(-1,1)
  
  N = np.arange(1, 7)
  models = [None for i in range(len(N))]

  for i in range(len(N)):

    models[i] = GaussianMixture(N[i], tol=1e-4,\
      max_iter=500,covariance_type='diag').fit(X)

  AIC = [m.aic(X) for m in models]

  print('fitted %s ...' % variable_name)
  
  best_aic = np.argmin(AIC)
  M_best = models[best_aic]
  n_comp = N[best_aic]

  x = np.linspace(
    np.min(values_for_variable), # ^^^ screwing up
    np.max(values_for_variable), 100000)
  logprob = M_best.score_samples(x.reshape(-1, 1))
  responsibilities = M_best.predict_proba(x.reshape(-1, 1))
  pdf = np.exp(logprob)
  pdf_individual = responsibilities * pdf[:, np.newaxis]

  if PLT_ALL:
    
    n_bins = freedman_diaconis_bins(X)

    ax1.hist(X, n_bins, density=True, histtype='stepfilled', color=PLOT_COLORS[0], alpha=0.4)
    x_range = (
      np.min(values_for_variable), 
      np.max(values_for_variable)
    )
    #sns.distplot(nn_values_for_variable, kde=False, norm_hist=True,
    #  hist_kws=hist_kws,ax=ax1)

    ax1.plot(x, pdf, '-k')
    ax1.plot(x, pdf_individual, '--k', lw=1)
    #ax1.text(0.04, 0.96, "Best-fit: %d comp (AIC = %2.f)" % (n_comp, best_aic),
    #    ha='left', va='top', transform=ax.transAxes,fontdict={'fontsize': 6})
   
    ax1.set_ylabel('$p(x)$',fontsize=6)
    ax1.set_xlim(np.min(values_for_variable)*0.9, np.max(values_for_variable)*1.1)
    ax1.set_ylim(0, np.max(pdf)*1.5)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
  
  if PLT_ALL:
    ax2.plot(N, AIC, color=PLOT_COLORS[3], lw=1, label='AIC')
    ax2.plot(N[best_aic], AIC[best_aic], \
      marker='o',color=PLOT_COLORS[0])
    ax2.annotate('N = %d' % (n_comp), \
      (N[best_aic], AIC[best_aic]),size=6)
    ax2.set_ylabel('AIC',fontsize=6)

    ax2.set_yticks([])
    #ax2.legend(prop={'size':6}, loc=2)

  return M_best

if PLT_ALL: 
  fig1, axes1 = plt.subplots(n_fig_rows, n_fig_cols, figsize=(15,15))
  fig2, axes2 = plt.subplots(n_fig_rows, n_fig_cols, figsize=(15,15))

transformed_values_for_variables = []
gmm_models = {}

for k in range(0, len(selected_variables)):
  variable_name = selected_variables[k]
  ax_title = str(variable_name)
  if PLT_ALL: 
    ax1 = axes1[int(k/n_fig_cols), k % n_fig_cols]
    ax2 = axes2[int(k/n_fig_cols), k % n_fig_cols]
    ax1.set_title(ax_title,fontdict={"fontsize":6})
    ax2.set_title(ax_title,fontdict={"fontsize":6})
  else:
    ax1,ax2 = None, None
  
  values_for_variable = positive_values_for_variables[k]
  gmm = fit_gmm(variable_name, values_for_variable, ax1, ax2)
  
  generated_values = gmm.sample(1000)[0][:,0]

  gmm_models[variable_name] = [gmm, generated_values]

PLT_ALL = False

if PLT_ALL: 

  top_title = 'Histogram of observed values, GMM estimation, and Gaussian components'

  fig1.suptitle(top_title, fontsize=8)
  fig1.tight_layout(rect=[0,0.03,0,1.25])
  fig1.subplots_adjust(hspace=1, wspace=0.35)

if PLT_ALL: 
  top_title = 'Optimization of number of components for each GMM based on AIC'
  
  fig2.suptitle(top_title, fontsize=8)
  fig2.tight_layout(rect=[0,0.03,0,1.25])
  fig2.subplots_adjust(hspace=1, wspace=0.35)
  
  plt.show()

n_synthetic = 200 * 5

synthetic_variable_values = np.zeros(
 (len(selected_variables), n_synthetic)
)

for k in range(0, len(selected_variables)):
  variable_name = selected_variables[k]
  gmm, generated_values = gmm_models[variable_name]
  synthetic_values_for_variable = generated_values[0:n_synthetic]
  synthetic_values_for_variable = np.asarray(synthetic_values_for_variable)
  synthetic_variable_values[k,:] = synthetic_values_for_variable

synthetic_variable_values = np.asarray(synthetic_variable_values)

imputed_variable_values = np.zeros(
 (len(selected_variables), len(selected_patient_ids) * n_time_points)
)

imputer = KNNImputer(n_neighbors=9)

num_column = 0
num_columns = imputed_variable_values.shape[1]

while num_column < num_columns:
  
  variable_value_to_impute = [
    values_for_variables[k][num_column] \
    for k in range(0, len(selected_variables))]
  
  for k in range(0, len(selected_variables)):
    v = variable_value_to_impute[k]
    if v is None or np.isnan(v):
      variable_value_to_impute[k] = None
    else:
      variable_value_to_impute[k] = \
        v - scaling_parameters[k][1]
  
  for k in range(0, len(selected_variables)):
    v = variable_value_to_impute[k]
    if v is None or np.isnan(v):
      variable_value_to_impute[k] = None
    else:
      variable_value_to_impute[k] = (
        boxcox(v+BOXCOX_A, lmbda=scaling_parameters[k][0]))
  
  variable_value_to_impute = \
    np.asarray(variable_value_to_impute)
  
  synthetic_variable_values[:,0] = variable_value_to_impute
  df = pd.DataFrame(synthetic_variable_values)
  
  imputed_vv = imputer.fit_transform(df.transpose()).transpose()
  imputed_variable_values[:,num_column] = imputed_vv[:,0]
  
  for x in imputed_vv[:,0]:
    if x is not None:
      if np.isnan(x):
        print('Found NAN for variable ' + str(k))
        print(variable_value_to_impute)
        print(imputed_vv[:,0])
        exit()

  num_column += 1
  print('imputing %d/%d' % (num_column, num_columns))

df = pd.DataFrame(imputed_variable_values)
patients_labels = []
patients_features = []

PLT_ALL= True

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
  patients_labels.append(patient_id in death_pt_ids)

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

y = np.asarray(patients_labels)
X = np.asarray([x.flatten() for x in patients_features])

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)

df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None

df_subset = df
X = df_subset[feat_cols].values
y = df_subset['y']

#n_components = 50
from sklearn.manifold import Isomap
pca_50 = Isomap(n_neighbors=5,n_components=15)

#pca_50 = PCA(n_components=n_components)
pca_result_50 = pca_50.fit_transform(X)
#print('Cumulative explained variation for %d principal components: %3.f' \
#  % (n_components, np.sum(pca_50.explained_variance_ratio_)))

X = pca_result_50
y = df['y']

def run_silhouhette_analysis():
  range_n_clusters = [2, 3, 4]

  for n_clusters in range_n_clusters:

    if PLT_ALL: 
      fig, (ax1, ax2) = plt.subplots(1, 2)
      fig.set_size_inches(18, 7)
      ax1.set_xlim([-0.1, 1])
      ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters)
    #clusterer = c.SpectralClustering(n_clusters=n_clusters, \
    #  eigen_solver='arpack', affinity='nearest_neighbors')
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
      ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

      ith_cluster_silhouette_values.sort()

      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      color = cm.nipy_spectral(float(i) / n_clusters)

      if PLT_ALL: 
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
          0, ith_cluster_silhouette_values,
          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      y_lower = y_upper + 10

    if PLT_ALL: 
      colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    
      ax1.set_title("Silhouette plot for the various clusters.")
      ax1.set_xlabel("Silhouette coefficient values")
      ax1.set_ylabel("Cluster label")
      ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
      ax1.set_yticks([])
      ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

      ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    if PLT_ALL: 
      ax2.set_title("Visualization of the clustered data.")
      ax2.set_xlabel("Feature space for the 1st feature")
      ax2.set_ylabel("Feature space for the 2nd feature")

      plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

  if PLT_ALL: plt.show()

run_silhouhette_analysis()

n_clusters = 3
cluster_sizes = [0 for i in range(0, n_clusters)]
cluster_data = [[], [], []]
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
colors = cm.rainbow(np.linspace(0, 1, n_clusters))

for i in range(0, X.shape[0]):
  cluster = int(kmeans.labels_[i])
  outcome = int(patients_labels[i])
  color = [colors[cluster]]
  marker = ['o', '+'][outcome]
  size = [2, 25][outcome]
  cluster_sizes[cluster] = cluster_sizes[cluster] + 1
  cluster_patient_data = np.zeros(len(selected_variables))
  for j in range(0, len(selected_variables)):
    m = np.mean(patients_features[i][j])
    cluster_patient_data[j] = m
  
  cluster_data[cluster].append(cluster_patient_data)
  if PLT_ALL: plt.scatter(X[i,0],X[i,1],c=color,marker=marker,s=size)

for i in range(0, X.shape[0]):
  cluster_num = int(kmeans.labels_[i])
  if cluster_num == cluster_sizes.index(np.min(cluster_sizes)):
    print(X[i])
    for j in range(0, len(selected_variables)):
      print(selected_variables[j], patients_features[i][j])

if PLT_ALL:
  plt.title('K-means clustering of 43 lab and vital sign variables in patients with COVID-19')
  plt.ylabel('PCA component #2')
  plt.xlabel('PCA component #1')

  patch1, patch2, patch3 = \
   mpatches.Patch(color=colors[0], label='Phenotype A'), \
   mpatches.Patch(color=colors[1], label='Phenotype B'), \
   mpatches.Patch(color=colors[2], label='Phenotype C')
  plt.legend(handles=[patch1,patch2,patch3], loc='lower right', title='Legend')

  plt.show()

cluster_datas = [np.asarray(c) for c in cluster_data]

means_cluster_data = [[], [], []]
stds_cluster_data = [[], [], []]

cluster_num = 0
for cluster_num in range(0, len(cluster_data)):

  cluster_data = cluster_datas[cluster_num]
  means_cluster_data[cluster_num] = np.zeros(len(selected_variables))
  for j in range(0, len(selected_variables)):
    v = [x[j] for x in cluster_data]
    m = np.mean(v)
    s = np.std(v)
    means_cluster_data[cluster_num][j] = m
  cluster_num += 1

means_cluster_data = np.asarray(means_cluster_data).T
stds_data = []

for j in range(0, len(selected_variables)):
  mean = np.mean(means_cluster_data[j])
  means_cluster_data[j] -= mean
  std = np.std(means_cluster_data[j])
  means_cluster_data[j] /= std

  #means_cluster_data[j] -= np.median(means_cluster_data[j])
  std = np.std(means_cluster_data[j])
  stds_data.append(std)

means_cluster_data = np.asarray([
  np.asarray(x) for x in means_cluster_data
])

stds_data = np.asarray(stds_data)

if PLT_ALL: plt.style.use('seaborn-paper')
df = pd.DataFrame(means_cluster_data, \
  columns=['Phenotype A', 'Phenotype B', 'Phenotype C'],
  index=selected_variables)

if PLT_ALL: fig, ax = plt.subplots()
ylocs = np.asarray(range(0, len(selected_variables))) * 1000

for cluster_num in range(0, n_clusters): 
  color = colors[cluster_num]
  if PLT_ALL: 
    markers, cap, bars = ax.errorbar(\
      means_cluster_data[:,cluster_num], \
     ylocs + cluster_num * 200, xerr=stds_data, \
     fmt='none', color=color)
    [bar.set_alpha(0.75) for bar in bars]

    ax.scatter(means_cluster_data[:,cluster_num], \
     ylocs + cluster_num * 200, marker='|', \
     s=5, color=color)
    ax.vlines(0, 0, ylocs[-1], alpha=0.35)

    plt.yticks(ylocs, labels=selected_variables, fontsize=6)

if PLT_ALL:
  patch1, patch2, patch3 = \
    mpatches.Patch(color=colors[0], label='Phenotype A'), \
    mpatches.Patch(color=colors[1], label='Phenotype B'),\
    mpatches.Patch(color=colors[2], label='Phenotype C')
  plt.legend(handles=[patch1,patch2,patch3], loc='lower right', title='Legend')

  #df.plot.barh(colormap=cm.rainbow, fontsize=6, capsize=4,xerr=stds_data)
  plt.title('Relative difference (%) between each phenotype and mean of phenotypes for all predictors', fontsize=8)
  plt.xlabel('Standardized difference with mean of all phenotypes for each value')
  plt.ylabel('Clinical predictors')
  plt.show()
exit()

if PLT_ALL: plt.scatter(pca_result_50[:, 0], pca_result_50[:, 1], c=kmeans.predict(data_subset), s=50, cmap='rainbow')
centers = kmeans.cluster_centers_
if PLT_ALL: plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
if PLT_ALL: plt.show()
