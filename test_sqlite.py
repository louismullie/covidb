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

def compare_labs_by(conn, col, val_pos, val_neg, lab_name, min_value=0, max_value=999999, query_tail=''):
  
  query = "SELECT lab_result_value from lab_data " + \
    " INNER JOIN patient_data ON " + \
    " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " lab_data.lab_name = '" + lab_name + "' AND " + \
    " lab_data.lab_result_status = 'resulted' AND " + \
    " patient_data." + col + " = "
  
  res = sql_fetch_all(conn, query + " '" + val_pos + "' " + query_tail)
  values_pos = [float(value[0]) for value in res]
  
  res = sql_fetch_all(conn, query + " '" +  val_neg + "' " + query_tail)
  values_neg = [float(value[0]) for value in res]
  
  plot_compare_kde(lab_name, col, val_pos, val_neg, values_pos, \
    values_neg, min_value, max_value)

def compare_labs_by_admit(conn, lab_name, min_value=0, max_value=999999):
  return compare_labs_by(conn, 'patient_was_admitted', 'yes', 'no', lab_name, \
                    min_value=min_value, max_value=max_value)

def compare_obs_by_admit(conn, obs_name, min_value=0, max_value=999999):
  return compare_obs_by(conn, 'patient_was_admitted', 'yes', 'no', obs_name, \
                    min_value=min_value, max_value=max_value)

def compare_labs_by_covid(conn, lab_name, min_value=0, max_value=999999):
  return compare_labs_by(conn, 'patient_covid_status', 'positive', 'negative', lab_name, \
                    min_value=min_value, max_value=max_value)

def compare_obs_by_covid(conn, obs_name, min_value=0, max_value=999999):
  return compare_obs_by(conn, 'patient_covid_status', 'positive', 'negative', obs_name, \
                    min_value=min_value, max_value=max_value)

def compare_labs_by_death(conn, lab_name, min_value=0, max_value=999999):
  return compare_labs_by(conn, 'patient_vital_status', 'dead', 'alive', lab_name, \
                    min_value=min_value, max_value=max_value, \
                    query_tail = 'AND patient_data.patient_covid_status = \'positive\'')

def compare_obs_by_death(conn, obs_name, min_value=0, max_value=999999):
  return compare_obs_by(conn, 'patient_vital_status', 'dead', 'alive', obs_name, \
                    min_value=min_value, max_value=max_value, \
                    query_tail = 'AND patient_data.patient_covid_status = \'positive\'')

def compare_obs_by(conn, col, val_pos, val_neg, obs_name, min_value=0, max_value=999999, query_tail=''):
  
  query = "SELECT observation_value from observation_data " + \
    " INNER JOIN patient_data ON " + \
    " observation_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " observation_data.observation_name = '"+obs_name+"' AND " + \
    " observation_data.observation_value NOT NULL AND " + \
    " patient_data." + col + " = "
  
  res = sql_fetch_all(conn, query + " '" + val_pos + "' " + query_tail)

  values_pos = [float(value[0]) for value in res]
  
  res = sql_fetch_all(conn, query + " '" + val_neg + "' " + query_tail)
  
  values_neg = [float(value[0]) for value in res]
  
  plot_compare_kde(obs_name, col, val_pos, val_neg, values_pos, \
    values_neg, min_value, max_value)

fig, axs = plt.subplots(5, 5)
fig.set_figheight(15)
fig.set_figwidth(15)

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

plt.sca(axs[0, 0])
compare_labs_by_covid(conn, 'lymphocyte_count', min_value=0.1, max_value=10)

plt.sca(axs[0, 1])
compare_labs_by_covid(conn, 'c_reactive_protein', min_value=50, max_value=350)

plt.sca(axs[0, 2])
compare_labs_by_covid(conn, 'd_dimer', min_value=100, max_value=6000)

plt.sca(axs[0, 3])
compare_labs_by_covid(conn, 'mean_platelet_volume', min_value=8, max_value=12)

plt.sca(axs[0, 4])
compare_labs_by_covid(conn, 'phosphate', min_value=0.5, max_value=2.5)

plt.sca(axs[1, 0])
compare_obs_by_covid(conn, 'fraction_inspired_oxygen', min_value=21, max_value=100)

plt.sca(axs[1, 1])
compare_obs_by_covid(conn, 'temperature', min_value=35, max_value=40.0)

plt.sca(axs[1, 2])
compare_obs_by_covid(conn, 'systolic_blood_pressure', min_value=50, max_value=200)

plt.sca(axs[1, 3])
compare_obs_by_covid(conn, 'heart_rate', min_value=40, max_value=180)

plt.sca(axs[1, 4])
compare_obs_by_covid(conn, 'oxygen_flow_rate', min_value=0, max_value=10)

plt.sca(axs[2, 0])
compare_labs_by_death(conn, 'lactic_acid', min_value=1, max_value=10)

plt.sca(axs[2, 1])
compare_labs_by_death(conn, 'pco2', min_value=35, max_value=100)

plt.sca(axs[2, 2])
compare_labs_by_death(conn, 'd_dimer', min_value=100, max_value=6000)

plt.sca(axs[2, 3])
compare_labs_by_death(conn, 'platelet_count', min_value=30, max_value=450)

plt.sca(axs[2, 4])
compare_labs_by_death(conn, 'alanine_aminotransferase', min_value=10, max_value=150)

plt.sca(axs[3, 0])
compare_obs_by_death(conn, 'temperature', min_value=36, max_value=39)

plt.sca(axs[3, 1])
compare_obs_by_death(conn, 'respiratory_rate', min_value=15, max_value=30)

plt.sca(axs[3, 2])
compare_obs_by_death(conn, 'systolic_blood_pressure', min_value=50, max_value=200)

plt.sca(axs[3, 3])
compare_obs_by_death(conn, 'heart_rate', min_value=40, max_value=180)

plt.sca(axs[3, 4])
compare_obs_by_death(conn, 'oxygen_flow_rate', min_value=0, max_value=10)

plt.sca(axs[4, 0])
compare_labs_by_admit(conn, 'creatinine', min_value=60, max_value=200)

plt.sca(axs[4, 1])
compare_labs_by_admit(conn, 'hemoglobin', min_value=50, max_value=140)

plt.sca(axs[4, 2])
compare_obs_by_admit(conn, 'respiratory_rate', min_value=15, max_value=30)

plt.sca(axs[4, 3])
compare_obs_by_admit(conn, 'heart_rate', min_value=40, max_value=180)

plt.sca(axs[4, 4])
compare_obs_by_admit(conn, 'systolic_blood_pressure', min_value=40, max_value=180)

plt.tight_layout()
plt.show()