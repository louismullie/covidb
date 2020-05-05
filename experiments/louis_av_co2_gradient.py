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

def compute_drug_odds_ratios(conn):

  query = "SELECT drug_name from drug_data"
  drugs = list(set([rec[0] for rec in sql_fetch_all(conn, query)]))

  odds_ratios = []
  for drug in drugs:
    if "'" in drug: continue
    odds = compare_drugs_by_death(conn, [drug])
    if odds[0] == -1: continue
    if odds[-1] < 5: continue
    odds_ratios.append([drug, odds])

  odds_ratios.sort(key=lambda x: x[1][0])

  for drug, odds in odds_ratios:
    print('%s - OR = %.2f (%.2f-%.2f), N exposed = %d' % \
         (drug, odds[0], odds[1], odds[2], odds[3]) )

def compare_drugs_by_death(conn, drug_names):
  
  query = "SELECT patient_site_uid from patient_data WHERE " + \
          "patient_covid_status = 'positive'"
  all_patients = set([rec[0] for rec in sql_fetch_all(conn, query)])

  query = "SELECT patient_site_uid from patient_data WHERE " + \
          "patient_covid_status = 'positive' AND patient_vital_status = 'dead'"
  dead_patients = set([rec[0] for rec in sql_fetch_all(conn, query)])
  alive_patients = all_patients.difference(dead_patients)

  query = "SELECT drug_data.patient_site_uid from drug_data INNER JOIN " + \
          " patient_data ON patient_data.patient_site_uid = " + \
          " drug_data.patient_site_uid WHERE " + \
          " patient_data.patient_covid_status = 'positive' AND " + \
          " drug_name IN ('" + "', '".join(drug_names) + "')"
  
  exposed_patients = set([rec[0] for rec in sql_fetch_all(conn, query)])
  
  nonexposed_patients = all_patients.difference(exposed_patients)

  a = len(list(exposed_patients & dead_patients))
  b = len(list(exposed_patients & alive_patients))
  c = len(list(nonexposed_patients & dead_patients))
  d = len(list(nonexposed_patients & alive_patients))
  
  if len(exposed_patients) == 0: return [-1, None, None, None]

  odds_ratio = (float(0.5+a)/float(0.5+c)) / \
               (float(0.5+b) / float(0.5+d))
  odds_ratio = round(odds_ratio, 2)


  log_or = log(odds_ratio)
  se_log_or = sqrt(1/float(0.5+a) + 1/float(0.5+b) + \
                   1/float(0.5+c) + 1/float(0.5+d))
  lower_ci = exp(log_or - 1.96 * se_log_or)
  upper_ci = exp(log_or + 1.96 * se_log_or)

  return [odds_ratio, lower_ci, upper_ci, len(exposed_patients)]

def summarize_variable(table, variable):

  query = "SELECT %s FROM %s " % (variable, table)

  if table == 'patient_data':
    query += "WHERE patient_covid_status = 'positive'"
  else:
    query += (" INNER JOIN patient_data ON " + \
    " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE "
    " patient_data.patient_covid_status = 'positive'")

  values = np.asarray([x[0] for x in sql_fetch_all(conn, query) if x[0] is not None])
  
  print("Summary for variable %s" % variable)
  print("Mean: %.2f" % np.mean(values))
  print("Std: %.2f" % np.std(values))

  pct_above70 = np.count_nonzero(values > 70) / len(values)
  print("Proportion above 70: %.2f" % pct_above70)

def correlate_labs(conn):
  
  query_art_pco2 = "SELECT patient_data.patient_site_uid, " + \
    " lab_sample_time, lab_result_value from lab_data " + \
    " INNER JOIN patient_data ON " + \
    " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " lab_data.lab_name = 'pco2' AND " + \
    " lab_data.lab_sample_site = 'arterial_blood' AND " + \
    " lab_data.lab_result_status = 'resulted' AND " + \
    " patient_data.patient_covid_status = 'positive'"

  query_ven_pco2 = "SELECT patient_data.patient_site_uid, " + \
    " lab_sample_time, lab_result_value from lab_data " + \
    " INNER JOIN patient_data ON " + \
    " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " lab_data.lab_name = 'pco2' AND " + \
    " lab_data.lab_sample_site = 'venous_blood' AND " + \
    " lab_data.lab_result_status = 'resulted' AND " + \
    " patient_data.patient_covid_status = 'positive'"

  query_d_dimer = "SELECT patient_data.patient_site_uid, " + \
    " lab_sample_time, lab_result_value from lab_data " + \
    " INNER JOIN patient_data ON " + \
    " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " lab_data.lab_name = 'd_dimer' AND " + \
    " lab_data.lab_result_status = 'resulted' AND " + \
    " patient_data.patient_covid_status = 'positive'"

  arterial_pco2s = sql_fetch_all(conn, query_art_pco2)
  venous_pco2s = sql_fetch_all(conn, query_ven_pco2)
  d_dimers =  sql_fetch_all(conn, query_d_dimer)
  
  pairs = []

  for arterial_pco2 in arterial_pco2s:
    id_1, st_1, rv_1 = arterial_pco2
    for venous_pco2 in venous_pco2s:
      id_2, st_2, rv_2 = venous_pco2
      if id_1 != id_2: continue
      av_pco2 = rv_2 - rv_1
      if av_pco2 < 0: continue
      dt12 = get_hours_between_datetimes(st_1, st_2)
      if dt12 < 2:
        pairs.append([arterial_pco2, venous_pco2])
  
  x = []
  y = []
  ids = []

  for d_dimer in d_dimers:
    id_1, st_1, rv_1 = d_dimer
    for arterial_pco2, venous_pco2 in pairs:
      id_2, st_2, rv_2 = arterial_pco2
      id_3, st_3, rv_3 = venous_pco2
      if not (id_1 == id_2 == id_3): continue
      td12 = get_hours_between_datetimes(st_1, st_2)
      td13 = get_hours_between_datetimes(st_1, st_3)
      if np.abs(td12) < 24 or np.abs(td13) < 24:
        x.append(rv_1)
        y.append(av_pco2)
        ids.append(id_1)

  print(len(x))
  print(len(np.unique(ids)))
  x = np.asarray(x)
  y = np.asarray(y)

  z = np.polyfit(x, y, 1)
  p = np.poly1d(z)

  plt.scatter(x, y)
  plt.plot(x, p(x), 'r--')
  plt.show()
  
  #print(arterial_pco2)
  #print(venous_pco2)
  #print(d_dimer)
  # plt.plot(arterial_pco2)
  # p#lt.plot(venous_pco2)
  # plt.plot(d_dimer)

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

#summarize_variable('patient_data', 'patient_age')
#exit()
correlate_labs(conn)
exit()
#exit()
#compute_drug_odds_ratios(conn)

#tabulate_column('patient_age', res, -3)
#tabulate_column('patient_sex', res, -2)
#tabulate_column('patient_covid_status', res, -4)
#tabulate_column('patient_death_status', res, -1)
