#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY
from postgresql_utils import sql_query, list_columns, list_tables
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_site_uid
from mappers import map_episode_unit_type

patient_mrns = []
patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

for row in patient_data_rows:
  patient_mrn = str(row[0])
  patient_mrns.append(patient_mrn)

# Get deaths
df = sql_query("SELECT DISTINCT * FROM dw_test.orcl_cichum_bendeces_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhredeces > '2020-01-01'")

diagnosis_data_rows = []

for index, row in df.iterrows():
  patient_mrn = str(row.dossier)

  diagnosis_type = 5
  diagnosis_time = row.dhredeces

  diagnosis_data_rows.append([
    patient_mrn, diagnosis_type, diagnosis_time
  ])

print('Total rows: %d' % len(diagnosis_data_rows))

write_csv(TABLE_COLUMNS['diagnosis_data'], diagnosis_data_rows, 
  os.path.join(CSV_DIRECTORY, 'diagnosis_data.csv'))