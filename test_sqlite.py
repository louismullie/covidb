import os, csv, sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from constants import SQLITE_DIRECTORY, CSV_DIRECTORY
from plot_utils import plot_compare_kde
from cli_utils import tabulate_column
from sqlite_utils import sql_fetch_all, sql_fetch_one

def compare_labs_by_covid(conn, lab_name, min_value=0, max_value=999999):
  return compare_labs_by(conn, 'patient_covid_status', '1', '2', lab_name, \
                    min_value=min_value, max_value=max_value)

def compare_obs_by_covid(conn, obs_name, min_value=0, max_value=999999):
  return compare_obs_by(conn, 'patient_covid_status', '1', '2', obs_name, \
                    min_value=min_value, max_value=max_value)

def compare_labs_by_death(conn, lab_name, min_value=0, max_value=999999):
  return compare_labs_by(conn, 'patient_death_status', '1', '2', lab_name, \
                    min_value=min_value, max_value=max_value, \
                    query_tail = ' AND patient_data.patient_covid_status = 1')

def compare_obs_by_death(conn, obs_name, min_value=0, max_value=999999):
  return compare_obs_by(conn, 'patient_death_status', '1', '2', obs_name, \
                    min_value=min_value, max_value=max_value, \
                    query_tail = ' AND patient_data.patient_covid_status = 1')

def compare_obs_by(conn, col, val_pos, val_neg, obs_name, min_value=0, max_value=999999, query_tail=''):
  
  query = "SELECT observation_value from observation_data " + \
    " INNER JOIN patient_data ON " + \
    " observation_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " observation_data.observation_name = '"+obs_name+"' AND " + \
    " patient_data." + col + " = "
  
  res = sql_fetch_all(conn, query + val_pos + query_tail)
  values_pos = [float(value[0]) for value in res]
  
  res = sql_fetch_all(conn, query + val_neg + query_tail)
  values_neg = [float(value[0]) for value in res]

  plot_compare_kde(obs_name, col, values_pos, \
    values_neg, min_value, max_value)

def compare_labs_by(conn, col, val_pos, val_neg, lab_name, min_value=0, max_value=999999, query_tail=''):
  
  query = "SELECT lab_result_value from lab_data " + \
    " INNER JOIN patient_data ON " + \
    " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " lab_data.lab_name = '" + lab_name + "' AND " + \
    " lab_data.lab_result_status = 1 AND " + \
    " patient_data." + col + " = "
  
  res = sql_fetch_all(conn, query + val_pos + query_tail)
  values_pos = [float(value[0]) for value in res]
  
  res = sql_fetch_all(conn, query + val_neg + query_tail)
  values_neg = [float(value[0]) for value in res]

  plot_compare_kde(lab_name, col, values_pos, \
    values_neg, min_value, max_value)

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

res = sql_fetch_all(conn, "SELECT * from patient_data")

tabulate_column('patient_age', res, -3)
tabulate_column('patient_sex', res, -2)
tabulate_column('patient_covid_status', res, -4)
tabulate_column('patient_death_status', res, -1)

#compare_by_covid(conn, 'Ferritine', min_value=5, max_value=10000)
#compare_by_covid(conn, 'Température', min_value=35, max_value=40)
#compare_by_covid(conn, 'Lympho #', max_value=10)
#compare_by_covid(conn, 'Phosphore', max_value=10)

#compare_by_covid(conn, 'VPM', max_value=18)
#compare_by_covid(conn, 'Protéine C Réac.', max_value=500)
#compare_by_covid(conn, 'Procalcitonine', max_value=100)
#compare_by_covid(conn, 'D-Dimère', max_value=50000)

compare_labs_by_death(conn, 'Lympho #')
compare_obs_by_death(conn, 'fraction_inspired_oxygen')

#query = "SELECT imaging_accession_uid from imaging_data " + \
#    " INNER JOIN patient_data ON " + \
#    " imaging_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
#    " patient_data.patient_covid_status = 2"

res = sql_fetch_all(conn, "SELECT * from patient_data WHERE patient_death_status=2")

#res = sql_fetch_all(conn, query)
#print(len(res))
#imaging_accession_uid = res[1]

# Fetch all imaging tests
res = sql_fetch_one(conn, "SELECT * from slice_data")
file = res[-13]

import numpy as np
pix = pd.read_csv(file).to_numpy()

from PIL import Image
dat = (pix / (np.max(pix)) * 255).astype(np.uint8)
im = Image.fromarray(dat)
im.save('test.jpeg')
