import os, csv, sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from math import log, sqrt, exp
from constants import SQLITE_DIRECTORY, CSV_DIRECTORY
from plot_utils import plot_compare_kde
from cli_utils import tabulate_column
from sqlite_utils import sql_fetch_all, sql_fetch_one
from time_utils import get_hours_between_datetimes

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.gaussian_process.kernels import DotProduct, \
  WhiteKernel, ConstantKernel, RBF

from sklearn.ensemble import RandomForestClassifier
from rf_regressor import RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import sklearn.cluster as c
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import shapiro


np.set_printoptions(precision=2, suppress=True)

def show_correlation_matrix(df):
  plt.matshow(df.corr())
  #plt.title('Correlations between laboratory parameters over time', y=-0.01)
  plt.xticks(range(0,len(feature_rows)), feature_rows, rotation='vertical',fontsize='6')
  plt.yticks(range(0,len(feature_rows)), feature_rows, fontsize='6')
  plt.gca().xaxis.set_ticks_position('top')
  plt.colorbar()

  plt.tight_layout(pad=2)
  plt.show()

def show_dist_plots(rows):
  nonnorm_vars = []
  fig, axes = plt.subplots(7, 7)
  k = 0
  print(rows)
  for k in range(0, np.min((49,len(rows)))):
    row = rows[k]
    nn_row = row[row != np.array(None)]
    pnorm = shapiro(nn_row)[1]
    if pnorm < 0.05:
      nonnorm_vars.append(feature_rows[k])
    lab = '%s, p = %.3f' % (feature_rows[k], pnorm)
    ax = axes[int(k/7), k % 7]
    ax.set_title(lab,fontdict={"fontsize":6})
    sns.distplot(nn_row, hist=False, kde_kws={"shade": True, "bw":1.5}, ax=ax)
    
    pct_missing = float(len(nn_row)) / float(len(row)) * 100
    print('Feature %s, pct. missing: %.2f, min: %.2f, max: %.2f' % 
      (feature_rows[k], pct_missing, np.min(nn_row), np.max(nn_row)))
    k += 1

  plt.setp(axes, yticks=[], xticks=[])
  plt.tight_layout()
  plt.show()

def run_imputation(input_array, random_state):

  input_array = np.asarray(input_array)
  
  rows = []

  for k in range(0, input_array.shape[1]):
    row = []
    for i in range(0, input_array.shape[0]):
      patient_data = input_array[i]
      row.extend(patient_data[k])
    row = np.asarray(row)
    rows.append(row)

  rows = np.asarray(rows)

  input_array = rows.transpose()
  input_shape = input_array.shape

  print('Input shape for imputer: %s' % str(input_shape))
  
  estimator = RandomForestRegressor(\
    n_estimators=4, random_state=random_state)

  imp_mean = IterativeImputer( \
  estimator=estimator,
   random_state=random_state)

  imp_mean.fit(input_array)
  imp_array = imp_mean.transform(input_array)
  input_array = input_array.transpose()

  df = pd.DataFrame(imp_array)

  imputed_array = imp_array.transpose()

  output_array = []
  for i in range(0, feature_shape[0]):
    rows = []
    for j in range(0, feature_shape[1]):
      rows.append(imputed_array[j,i*3:i*3+3])
    output_array.append(rows)
  output_array = np.asarray(output_array)
  return output_array


db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

query_info = "SELECT patient_site_uid, patient_age, patient_sex from patient_data WHERE " + \
        " patient_data.patient_covid_status = 'positive'"

#pt_infos = dict((str(x[0]), [map_age(x[1]), map_sex(x[2])]) for x in sql_fetch_all(conn, query_info))

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
         " episode_data.episode_unit_type = 'intensive_care_unit' AND " + \
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

query = "SELECT lab_data.patient_site_uid, lab_name, lab_sample_time, lab_result_value from lab_data " + \
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

full_lab_data = [[str(value[0]), str(value[1]), str(value[2]), float(value[3])] for value in res1] # + \
           #[[str(value[0]), str(value[1]), str(value[2]), float(value[3])] for value in res2]

full_lab_bins = {}
full_lab_names = []
full_patients_with_data = {}
patient_num = 0

for patient_id, lab_name, lab_sample_time, lab_value in full_lab_data:
  if patient_id not in eligible_patients: continue
  full_lab_names.append(lab_name)

  if patient_id not in full_lab_bins:
    full_lab_bins[patient_id] = {}
  if lab_name not in full_patients_with_data:
    full_patients_with_data[lab_name] = []
  if lab_name not in full_lab_bins[patient_id]:
    full_lab_bins[patient_id][lab_name] = []
  
  for episode_start_time in eligible_episodes[patient_id]:
    hours_since_admission = get_hours_between_datetimes(episode_start_time, lab_sample_time)
    full_lab_bins[patient_id][lab_name].append([hours_since_admission, lab_value])
    full_patients_with_data[lab_name].append(patient_id)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from sklearn.pipeline import make_pipeline

num_labs_per_fig = 5
n_time_points = 3
lab_data = []

full_lab_names = list(set(full_lab_names))

regressed_data = {}
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import Lasso

for patient_id in np.unique(full_patients_with_data['creatinine']):
  
  if patient_id not in eligible_patients: continue 

  if True: fig, axs = plt.subplots(num_labs_per_fig,1, constrained_layout=True)
  
  if patient_id not in regressed_data:
    regressed_data[patient_id] = {}

  variable_num = 0
  num_labs_plotted = 0

  for variable_num in range(0, len(full_lab_names)):

    if True:
      if num_labs_plotted >= num_labs_per_fig:
        ax = axs[num_labs_per_fig-1]
      else:
        ax = axs[num_labs_plotted]

    variable_name = full_lab_names[variable_num]
    variable_num += 1

    if variable_name not in full_lab_bins[patient_id]: 
      continue

    ex_labs = full_lab_bins[patient_id][variable_name]
    ex_labs.sort(key=lambda x: x[0])
    ex_labs = np.asarray(ex_labs) 

    np.random.seed(1)
    labs_X = np.asarray(ex_labs[:,0])
    
    labs_X -= np.min(labs_X)
    labs_X /= 24

    labs_y = np.abs(ex_labs[:,1])

    if len(labs_X) < 2: 
      
      for k in range(0,n_time_points):
        lab_data.append([patient_id, variable_name, k, labs_y[0]])
      
      continue

    num_labs_plotted += 1
    
    X = np.atleast_2d(labs_X).T
    y = labs_y.ravel()

    dx = np.max(X) - np.min(X)
    x = np.atleast_2d(np.linspace(0, 3, n_time_points * 24)).T

    m = np.mean(labs_X)
    gp = Lasso(alpha=1)
    gp.fit(X, y)

    kernel = C(0.1, (1e-23, 1e5)) * \
      RBF(0.1, (1e-23, 1e10)) + \
      WhiteKernel(0.1, (1e-23, 1e5))
  
    Xy_pairs = [[labs_X[k], labs_y[k]] for k in range(0,len(labs_X))]
  
    for i in range(0, int(np.max(X))):
      Xy_pairs.append([i,gp.predict([[i]])[0]])
  
    Xy_pairs.sort(key=lambda x: x[0])
    Xy_pairs = np.asarray(Xy_pairs) 

    labs_X = Xy_pairs[:,0]
    labs_y = np.abs(Xy_pairs[:,1])

    X2 = np.atleast_2d(labs_X).T
    y2 = labs_y.ravel()

    gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-6)
    gp2.fit(X2, y2)

    #gp = make_pipeline(PolynomialFeatures(degree=10), Ridge())
    #gp.fit(X,y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred_lasso = gp.predict(x)
    y_pred, sigma = gp2.predict(x,return_std=True)

    for k in range(0,n_time_points):
      lab_data.append([patient_id, variable_name, k, y_pred[k*24]])
    if True:
      p1 = ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600,
                        (y_pred + 1.9600)[::-1]]),
         alpha=.25, fc='b', ec='None', label='95% confidence interval')

      p2 = ax.plot(X2, y2, 'g.', markersize=10, label='Added points')
      p3 = ax.plot(X, y, 'k.', markersize=10, label='Observations')
      p4 = ax.plot(x, y_pred, 'r:', label='Prediction')
      #p5 = ax.plot(x, y_pred_lasso, 'r:', label='Lasso prediction')

      ax.set_xlim(0, n_time_points)
      ax.set_xlabel('Number of days since beginning of episode')
      ax.set_ylabel(variable_name)

      ax.legend([(p1[0],p4[0]), p2[0], p3[0], p4[0]], ['95% CI', 'Added points',\
    'Observed points', 'Prediction from full data'], loc='upper right')

      ax.set_title(variable_name + ' over %.2f days' % n_time_points)

    #if num_labs_plotted == num_labs_per_fig: break
  patient_num += 1
  if True: plt.show()
  print('----------------- %s ------------------------------------' % str(patient_num))
  if patient_num > 1: break

import pickle
#pickle.dump(lab_data, open('regressed_data.sav', 'wb'))
lab_data = pickle.load(open('regressed_data.sav', 'rb'))

lab_bins = {}
lab_names = set()
patients_with_data = {}

for patient_id, lab_name, lab_sample_time, lab_value in lab_data:
  if patient_id not in eligible_patients: continue
  lab_names.add(lab_name)
  if patient_id not in lab_bins:
    lab_bins[patient_id] = {}
  if lab_name not in patients_with_data:
    patients_with_data[lab_name] = set()
  if lab_name not in lab_bins[patient_id]:
    lab_bins[patient_id][lab_name] = [None, None, None]
  
  if lab_sample_time == 0:
    lab_bins[patient_id][lab_name][0] = lab_value
  elif lab_sample_time == 1:
    lab_bins[patient_id][lab_name][1] = lab_value
  elif lab_sample_time == 2:
    lab_bins[patient_id][lab_name][2] = lab_value
  else: continue
  
  patients_with_data[lab_name].add(patient_id)

feature_array = []
patient_array = []
complete_lab_names = set()

patient_num = 0
NONNORM_VARS = ['creatinine', 'alkaline_phosphatase']

OTHER_VARS = ['alanine_aminotransferase', 'alkaline_phosphatase', 
'base_excess', 'basophil_count', 'bicarbonate', 'c_reactive_protein', 
'chloride', 'creatinine', 'diastolic_blood_pressure', 'eosinophil_count', 
'fraction_inspired_oxygen', 'glucose', 'heart_rate', 'hemoglobin', 
'lymphocyte_count', 'mean_corpuscular_hemoglobin', 'mean_corpuscular_volume', 
'mean_platelet_volume', 'monocyte_count', 'neutrophil_count', 'o2_sat', 
'oxygen_flow_rate', 'oxygen_saturation', 'pco2', 'phosphate', 'platelet_count', 
'po2', 'potassium', 'procalcitonin', 'red_blood_cell_count', 'respiratory_rate', 
'sodium', 'systolic_blood_pressure', 'temperature', 'total_bilirubin', 
'white_blood_cell_count']

deltas = {}

PATIENT_COMPLETENESS_THRESHOLD = 0.05
LAB_COMPLETENESS_THRESHOLD = 0.02

for patient_id in sorted(lab_bins):
  feature_rows = []
  feature_subarray = []
 
  num_labs_with_data = 0.0
  for lab_name in sorted(lab_names):
    
    pct_with_data = len(patients_with_data[lab_name]) / len(eligible_patients)
    if pct_with_data > LAB_COMPLETENESS_THRESHOLD:
      complete_lab_names.add(lab_name)
      feature_rows.append(lab_name)
      if lab_name not in lab_bins[patient_id]: 
        feature_subarray.append(np.asarray([None, None, None]))
      else:
        lab_values = lab_bins[patient_id][lab_name]
        if lab_name in NONNORM_VARS:
          lab_values = [np.log(x+0.001) if x != None \
            else None for x in lab_values]
        num_labs_with_data += np.count_nonzero(lab_values != None)
        
        a, b, c = lab_values

        if a is not None: a = np.abs(a)
        if b is not None: b = np.abs(b)
        if c is not None: c = np.abs(c)

        feature_subarray.append([a,b,c])

        if lab_name not in deltas:
          deltas[lab_name] = []
        if a is not None and b is not None:
          deltas[lab_name].append(np.abs(b - a))
        if c is not None and b is not None:
          deltas[lab_name].append(np.abs(c - b))
  
  if len(complete_lab_names) == 0:
    pct_labs_with_data = 0
  else:
    pct_labs_with_data = num_labs_with_data / len(complete_lab_names) / 3
  
  if pct_labs_with_data > PATIENT_COMPLETENESS_THRESHOLD:
    feature_array.append(feature_subarray)
    patient_array.append(patient_id)

  patient_num += 1

print('Total %d patients' % len(patient_array))
print('Total %d predictors' % len(feature_rows))

# Impute the mean for patients having no values for a test
patient_num = 0
for feature_subarray in feature_array:
  lab_num = 0
  for lab_name in complete_lab_names:
    if len(feature_subarray[lab_num != None]) == 0:
      v = [feature_array[fs_num][lab_num] \
        for fs_num in range(0, len(feature_array))]
      v = np.asarray(v).flatten()
      x = np.mean([np.abs(y) for y in v if y != None])
      feature_array[patient_num][lab_num] = [x, x, x]
    
    lab_num += 1
  patient_num += 1

feature_array = np.asarray(feature_array)
feature_shape = feature_array.shape

output_array = feature_array #run_imputation(feature_array, 0)
all_pt_ids = []
labels = []
num_patient = 0
for patient_id in sorted(patient_array):
  all_pt_ids.append(patient_id)
  #crp = imp_array[num_patient][feature_rows.index('c_reactive_protein')]
  #pct = imp_array[num_patient][feature_rows.index('white_blood_cell_count')]
  #inflammatory_subtype = pct > 10
  #labels.append(inflammatory_subtype)
  labels.append( (patient_id in death_pt_ids ) )
  num_patient += 1

output_array = run_imputation(output_array, 0)

feature_array = output_array

rows = []

for k in range(0, feature_array.shape[1]):
  row = []
  for i in range(0, feature_array.shape[0]):
    a,b,c = feature_array[i,k]
    row.extend([a,b,c])
  row = np.asarray(row)
  rows.append(row)

rows = np.asarray(rows)
show_dist_plots(rows)


y = np.asarray(labels)
X = np.asarray([x.flatten() for x in output_array])

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
#show_correlation_matrix(df)

df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None

df_subset = df
X = df_subset[feat_cols].values
y = df_subset['y']

print(X)
#pca_50 = GaussianRandomProjection(eps=0.9, random_state=rng)
#pca_result_50 = pca_50.fit_transform(X)
n_components = 8

pca_50 = PCA(n_components=n_components)
pca_result_50 = pca_50.fit_transform(X)
print('Cumulative explained variation for %d principal components: %3.f' \
  % (n_components, np.sum(pca_50.explained_variance_ratio_)))

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
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot for the various clusters.")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # Average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    #ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #            c="white", alpha=1, s=200, edgecolor='k')

    #for i, c in enumerate(centers):
    #    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                s=50, edgecolor='k')

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
  outcome = int(labels[i])
  color = [colors[cluster]]
  marker = ['o', '+'][outcome]
  size = [2, 25][outcome]
  cluster_sizes[cluster] = cluster_sizes[cluster] + 1
  cluster_patient_data = np.zeros(len(feature_rows))
  for j in range(0, len(feature_rows)):
    m = np.mean(output_array[i][j])
    cluster_patient_data[j] = m
  
  cluster_data[cluster].append(cluster_patient_data)
  plt.scatter(X[i,0],X[i,1],c=color,marker=marker,s=size)

for i in range(0, X.shape[0]):
  cluster_num = int(kmeans.labels_[i])
  if cluster_num == cluster_sizes.index(np.min(cluster_sizes)):
    print(X[i])
    print(feature_rows)
    print(output_array[i])

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
#print(cluster_data)
means_cluster_data = [[], [], []]
stds_cluster_data = [[], [], []]

cluster_num = 0
for cluster_num in range(0, len(cluster_data)):

  cluster_data = cluster_datas[cluster_num]
  means_cluster_data[cluster_num] = np.zeros(len(feature_rows))
  for j in range(0, len(feature_rows)):
    v = [x[j] for x in cluster_data]
    m = np.mean(v)
    s = np.std(v)
    means_cluster_data[cluster_num][j] = m
  cluster_num += 1

means_cluster_data = np.asarray(means_cluster_data).T
stds_data = []

for j in range(0, len(feature_rows)):
  # Export as percentage change vs. mean
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
  index=feature_rows)

fig, ax = plt.subplots()
ylocs = np.asarray(range(0, len(feature_rows))) * 1000

for cluster_num in range(0, n_clusters): 
  color = colors[cluster_num]
  markers, cap, bars = ax.errorbar(means_cluster_data[:,cluster_num], \
   ylocs + cluster_num * 200, xerr=stds_data, fmt='none', color=color)
  [bar.set_alpha(0.75) for bar in bars]
  ax.scatter(means_cluster_data[:,cluster_num], \
   ylocs + cluster_num * 200, marker='|', s=5, color=color)
ax.vlines(0, 0, ylocs[-1], alpha=0.35)
plt.yticks(ylocs, labels=feature_rows, fontsize=6)

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
