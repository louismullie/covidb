#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY, DRUG_SKIP_VALUES
from postgresql_utils import sql_query
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid
from mappers import map_time, map_drug_name, map_drug_route, map_drug_frequency

row_count = 0
patient_data_rows = []
patient_mrns = []
patient_covid_statuses = {}
pcr_sample_times = {}

patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

for row in patient_data_rows:
  patient_mrn = str(row[0])
  patient_mrns.append(patient_mrn)
  pcr_sample_times[patient_mrn] = row[2]

df = sql_query("SELECT * from dw_test.orcl_hev_bipap_live WHERE " + \
  "start_dtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

intervention_data_rows = []

for index, row in df.iterrows():
  
  patient_mrn = str(int(row.dossier))
  intervention_start_time = str(row.start_dtm)
  intervention_end_time = str(row.end_dtm)

  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[patient_mrn], intervention_start_time)
  
  if delta_hours < -48: continue

  intervention_data_rows.append([
    patient_mrn, 
    'mechanical_ventilation',
    intervention_start_time,
    intervention_end_time
  ])

print('Total rows: %d' % len(intervention_data_rows))

write_csv(TABLE_COLUMNS['intervention_data'], intervention_data_rows, 
  os.path.join(CSV_DIRECTORY, 'intervention_data.csv'))