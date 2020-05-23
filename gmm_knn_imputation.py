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
  plt.plot(points[:,0],points[:,1], label=ylabels[k], \
    marker='o', color=PLOT_COLORS[k])

plt.title("Missing data for different sampling strategies")
plt.ylabel("Number of features exceeding threshold")
plt.xlabel("Fraction of values that are not missing")

plt.legend()
plt.show()

xy_data = percentages[:,means>0.25]
xlabels = lab_names[means > 0.25]
n_plotted = len(lab_names)

fig, ax = plt.subplots()

im = ax.matshow(xy_data)
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

threshold_missingness = 0.25
percentages = percentages[8,:]

percentages_thresh = percentages > threshold_missingness
num_vars_left = np.count_nonzero(percentages_thresh)
selected_variables = lab_names[percentages > threshold_missingness]
print('Total %d variables remaining' % num_vars_left)
print(selected_variables)

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

fig, ax = plt.subplots()

im = ax.matshow(pct_missing_timepoints)

time_points_labels = ['t=%dh' % \
  (i * hours_per_period + left_limit) \
  for i in range(0, n_time_points)]

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

for variable_name in selected_variables:
  values_for_variable = []
  for patient_id in selected_patient_ids:
    if variable_name in lab_bins[patient_id]:
      v = lab_bins[patient_id][variable_name]
      values_for_variable.extend(v)
    else:
      values_for_variable.extend([None for i in range(0,n_time_points)])
  values_for_variables.append(values_for_variable)

n_fig_rows = 7
n_fig_cols = 6

n_fig_total = n_fig_rows * n_fig_cols

if len(values_for_variables) > n_fig_total:
  print('Warning: not all variables plotted')

fig, axes = plt.subplots(n_fig_rows, n_fig_cols, figsize=(15,15))
#fig.suptitle('Distribution of selected variables ' + \
#  'and p-value for Shapiro\'s test of normality')

positive_values_for_variables = []

for k in range(0, len(selected_variables)):
  variable_name = selected_variables[k]
  values_for_variable = values_for_variables[k]
  
  nn_values_for_variable = np.asarray([v for \
    v in values_for_variable if v is not None])
  min_value = np.min(nn_values_for_variable)
  nn_values_for_variable = np.asarray([v - min_value  \
   for v in nn_values_for_variable])
  positive_values_for_variables.append(nn_values_for_variable)

  p_is_not_norm = shapiro(nn_values_for_variable)[1]
  
  ax_title = '%s, p = %.4f' % (variable_name, p_is_not_norm)
  
  ax = axes[int(k/n_fig_cols), k % n_fig_cols]
  
  ax.set_title(ax_title,fontdict={"fontsize":5})
  sns.distplot(nn_values_for_variable, kde_kws={ \
    'shade': False, 'color': '#000000','bw':'scott' \
  }, hist_kws={'color': PLOT_COLORS[0], 'alpha':0.4},ax=ax)
  
  ax.set_ylabel('$p(x)$',fontsize=6)

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

plt.setp(axes, yticks=[], xticks=[])
plt.tight_layout()
plt.show()

labels = [None for k in range(0, n_fig_total)]

BOXCOX_A = 1.5

from sklearn.mixture import GaussianMixture

def fit_gmm(variable_name,values_for_variable,ax1, ax2,show_plot=True):
  
  random_state = np.random.RandomState(seed=1)
  X = values_for_variable.reshape(-1,1)
  
  N = np.arange(1, 7)
  models = [None for i in range(len(N))]

  for i in range(len(N)):

    models[i] = GaussianMixture(N[i], tol=1e-4,\
      max_iter=500,covariance_type='spherical').fit(X)

  AIC = [m.aic(X) for m in models]

  print('fitted %s ...' % variable_name)
  
  best_aic = np.argmin(AIC)
  M_best = models[best_aic]
  n_comp = N[best_aic]

  x = np.linspace(0, len(values_for_variable), 100000)
  logprob = M_best.score_samples(x.reshape(-1, 1))
  responsibilities = M_best.predict_proba(x.reshape(-1, 1))
  pdf = np.exp(logprob)
  pdf_individual = responsibilities * pdf[:, np.newaxis]

  if show_plot:
    ax1.hist(X, 30, density=True, histtype='stepfilled', color=PLOT_COLORS[0], alpha=0.4)
    ax1.plot(x, pdf, '-k')
    ax1.plot(x, pdf_individual, '--k', lw=1)
    ax1.text(0.04, 0.96, "Best-fit: %d comp (AIC = %2.f)" % (n_comp, best_aic),
        ha='left', va='top', transform=ax.transAxes,fontdict={'fontsize': 6})
   
    ax1.set_ylabel('$p(x)$',fontsize=6)
    ax1.set_xlim(0, np.max(X))
    ax1.set_ylim(0, np.max(pdf)*1.2)
    ax1.set_xticks([])
    ax1.set_yticks([])
  
    # plot 2: AIC and BIC
    ax2.plot(N, AIC, color=PLOT_COLORS[3], lw=1, label='AIC')
    ax2.plot(N[best_aic], AIC[best_aic], \
      marker='o',color=PLOT_COLORS[0])
    ax2.annotate('N = %d' % (n_comp), \
      (N[best_aic], AIC[best_aic]),size=6)
    ax2.set_ylabel('AIC',fontsize=6)

    ax2.set_xticks([])
    ax2.set_yticks([])
    #ax2.legend(prop={'size':6}, loc=2)

  return M_best

fig1, axes1 = plt.subplots(n_fig_rows, n_fig_cols, figsize=(15,15))
fig2, axes2 = plt.subplots(n_fig_rows, n_fig_cols, figsize=(15,15))

transformed_values_for_variables = []
gmm_models = {}

for k in range(0, len(selected_variables)):
  variable_name = selected_variables[k]
  ax_title = str(variable_name)
  ax1 = axes1[int(k/n_fig_cols), k % n_fig_cols]
  ax2 = axes2[int(k/n_fig_cols), k % n_fig_cols]
  
  ax1.set_title(ax_title,fontdict={"fontsize":6})
  ax2.set_title(ax_title,fontdict={"fontsize":6})
  min_val = np.min([i for i in values_for_variables[k] if i is not None])
  vs = positive_values_for_variables[k]
  values_for_variable = vs
  values_for_variable, lmbda = boxcox(np.asarray(vs)+1)
  gmm = fit_gmm(variable_name, values_for_variable, ax1,ax2)
  
  generated_values = gmm.sample(1000)[0]
  generated_values = inv_boxcox(
    np.asarray(generated_values[:,0]), lmbda) - 1
  generated_values = [v + min_val \
    for v in generated_values]
  generated_values = np.asarray(generated_values)

  gmm_models[variable_name] = [gmm, generated_values]

#plt.title('Transformed distribution of selected ' + \
#  'variables and p-value for Shapiro\'s test of normality', fontsize=7)

plt.setp(axes1, yticks=[], xticks=[])
plt.show()

#plt.title('Transformed distribution of selected ' + \
#  'variables and p-value for Shapiro\'s test of normality', fontsize=7)

plt.setp(axes2, yticks=[], xticks=[])
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

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

num_column = 0
num_columns = imputed_variable_values.shape[1]

while num_column < num_columns:
  variable_value_to_impute = [values_for_variables[k][num_column] \
    for k in range(0, len(selected_variables))]
  variable_value_to_impute = np.asarray(variable_value_to_impute)
  synthetic_variable_values[:,0] = variable_value_to_impute
  df = pd.DataFrame(synthetic_variable_values)

  imputed_df = imputer.fit_transform(df.transpose()).transpose()
  imputed_variable_values[:,num_column] = imputed_df[:,0]
  
  num_column += 1
  print('imputing %d/%d' % (num_column, num_columns))

df = pd.DataFrame(imputed_variable_values)
patients_labels = []
patients_features = []

for i in range(0, len(selected_patient_ids)):

  patient_id = selected_patient_ids[i]
  patient_features = []

  for k in range(0, len(selected_variables)):
    variable_name = selected_variables[k]
    imputed_values_for_variable = df.values[k]
    original_values_for_variable = values_for_variables[k]
   
    j = i*n_time_points
    orig_pt_var = original_values_for_variable[j:j+n_time_points]
    imput_pt_var = imputed_values_for_variable[j:j+n_time_points]
    print(variable_name, orig_pt_var[1:4], imput_pt_var[1:4])
    
    patient_features.append(imput_pt_var[1:4])

  patients_features.append(np.asarray(patient_features))
  patients_labels.append(patient_id in death_pt_ids)

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
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("Silhouette plot for the various clusters.")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ax2.set_title("Visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

  plt.show()

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
  plt.scatter(X[i,0],X[i,1],c=color,marker=marker,s=size)

for i in range(0, X.shape[0]):
  cluster_num = int(kmeans.labels_[i])
  if cluster_num == cluster_sizes.index(np.min(cluster_sizes)):
    print(X[i])
    for j in range(0, len(selected_variables)):
      print(selected_variables[j], patients_features[i][j])

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

plt.style.use('seaborn-paper')
df = pd.DataFrame(means_cluster_data, \
  columns=['Phenotype A', 'Phenotype B', 'Phenotype C'],
  index=selected_variables)

fig, ax = plt.subplots()
ylocs = np.asarray(range(0, len(selected_variables))) * 1000

for cluster_num in range(0, n_clusters): 
  color = colors[cluster_num]
  markers, cap, bars = ax.errorbar(means_cluster_data[:,cluster_num], \
   ylocs + cluster_num * 200, xerr=stds_data, fmt='none', color=color)
  [bar.set_alpha(0.75) for bar in bars]
  ax.scatter(means_cluster_data[:,cluster_num], \
   ylocs + cluster_num * 200, marker='|', s=5, color=color)
ax.vlines(0, 0, ylocs[-1], alpha=0.35)
plt.yticks(ylocs, labels=selected_variables, fontsize=6)

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

plt.scatter(pca_result_50[:, 0], pca_result_50[:, 1], c=kmeans.predict(data_subset), s=50, cmap='rainbow')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
