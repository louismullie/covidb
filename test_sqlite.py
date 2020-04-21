import os, csv, sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from constants import SQLITE_DIRECTORY, CSV_DIRECTORY
from plot_utils import plot_compare_kde
from cli_utils import tabulate_column
from sqlite_utils import sql_fetch_all, sql_fetch_one

def compare_by_covid(conn, lab_name, min_value=0, max_value=999999):
  
  query = "SELECT lab_result_value from lab_data " + \
    " INNER JOIN patient_data ON " + \
    " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " lab_data.lab_name = '" + lab_name + "' AND " + \
    " lab_data.lab_result_status = 1 AND " + \
    " patient_data.patient_covid_status = "

  res = sql_fetch_all(conn, query + "1")
  values_pos = [float(value[0]) for value in res]

  res = sql_fetch_all(conn, query + "2")
  values_neg = [float(value[0]) for value in res]

  plot_compare_kde(lab_name, 'COVID', values_pos, \
    values_neg, min_value, max_value)

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

res = sql_fetch_all(conn, "SELECT * from patient_data")

tabulate_column('patient_covid_status', res, -3)
tabulate_column('patient_age', res, -2)
tabulate_column('patient_sex', res, -1)

#compare_by_covid(conn, 'Ferritine', min_value=5, max_value=10000)
#compare_by_covid(conn, 'Température', min_value=35, max_value=40)
#compare_by_covid(conn, 'Lympho #', max_value=10)
#compare_by_covid(conn, 'Phosphore', max_value=10)

#compare_by_covid(conn, 'VPM', max_value=18)
#compare_by_covid(conn, 'Protéine C Réac.', max_value=500)
#compare_by_covid(conn, 'Procalcitonine', max_value=100)
#compare_by_covid(conn, 'D-Dimère', max_value=50000)

#query = "SELECT imaging_accession_uid from imaging_data " + \
#    " INNER JOIN patient_data ON " + \
#    " imaging_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
#    " patient_data.patient_covid_status = 2"

query = "SELECT imaging_accession_uid from imaging_data " + \
    " INNER JOIN patient_data ON " + \
    " imaging_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
    " patient_data.patient_covid_status = 2"

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
