#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv
import numpy as np
import pandas as pd

from constants import DEBUG, COLUMNS, LIVE_SHEET_FILENAME
from sql_utils import sql_query, list_columns, list_tables
from file_utils import write_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid
from mappers import map_lab_sample_site

row_count = 0
patient_data_rows = []
patient_mrns = []
pcr_sample_times = {}

reader = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

for row in reader:
  if not row_count == 0:
    patient_data_rows.append(row)
    patient_mrn = row[0]
    patient_mrns.append(patient_mrn)
    pcr_sample_times[str(patient_mrn)] = row[1]
  row_count += 1

#list_columns('oacis_lb')

#df = sql_query("SELECT * FROM dw_v01.cerner_labs_table WHERE perform_dt_tm > 2020-01-01 AND person_id in (" + ", ".join(patient_mrns) + ") LIMIT 100")
df = sql_query("SELECT * FROM dw_v01.oacis_lb WHERE " +
    "lbres_ck IS NOT NULL AND resultunit IS NOT NULL AND resultdtm IS NOT NULL AND " +
    "specimencollectiondtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

lab_data_rows = []

for index, row in df.iterrows():
  
  patient_mrn = row.dossier
  lab_name = row.longdesc
  lab_sample_site = row.specimencollectionmethodcd
  lab_sample_time = row.specimencollectiondtm
  lab_result_time = row.resultdtm
  lab_result_value = row.lbres_ck
  lab_result_units = row.resultunit

  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[str(patient_mrn)], str(lab_sample_time))

  if delta_hours < 24:
    lab_data_rows.append([
      patient_mrn, lab_name, 
      map_lab_sample_site(lab_sample_site), lab_sample_time, 
      lab_result_time, lab_result_value, lab_result_units
    ])

print('Total rows: %d' % len(lab_data_rows))
write_csv(COLUMNS['lab_data'], lab_data_rows, './csv/lab_data.csv')