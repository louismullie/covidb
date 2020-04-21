#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv, re, os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, LIVE_SHEET_FILENAME, CSV_DIRECTORY
from time_utils import get_datetime_seconds, get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid, generate_accession_uid
from file_utils import write_csv, read_csv
from postgresql_utils import sql_query

from mappers import map_patient_ramq, map_patient_covid_status, map_patient_age, map_patient_sex

live_sheet_rows = read_csv(LIVE_SHEET_FILENAME, remove_duplicates=True)

print('Number of rows in live sheet: %d' % len(live_sheet_rows))
patient_data = {}
patient_mrns = []
patient_covid_statuses = {}

for row in live_sheet_rows:

  is_external = (row[-3] == 'External')
  if is_external: continue

  patient_mrn = str(row[0])
  patient_ramq = str(row[1])
  patient_age = row[-1]
  patient_birth_sex = row[-2]
  patient_covid_status = map_patient_covid_status(row[-4])
  
  pcr_result_time = row[-6]
  pcr_sample_time = row[-7]
  
  if patient_mrn not in patient_covid_statuses:
    patient_covid_statuses[patient_mrn] = [patient_covid_status]
  else:
    patient_covid_statuses[patient_mrn].append(patient_covid_status)

  try:
    if patient_mrn not in patient_data:
      patient_data[patient_mrn] = []

    patient_data[patient_mrn].append([
      patient_mrn,
      patient_ramq,
      pcr_sample_time,
      'CHUM',
      '',
      patient_covid_status,
      map_patient_age(patient_age),
      map_patient_sex(patient_birth_sex)
    ])

  except Exception as err:

    if DEBUG:
      print('Skipping row: %s' % row)
      print('  due to error: %s' % err)
    continue

  patient_mrns.append(patient_mrn)

filtered_patient_data = []

# Build a list of unique patients
for patient_mrn in patient_data:

  patient_rows = patient_data[str(patient_mrn)]
  patient_rows.sort(key=lambda x: get_datetime_seconds(x[2]))
  covid_status = min(patient_covid_statuses[patient_mrn])
  found_positive = False
  
  # Choose the first PCR entry - to be reviseds
  if covid_status > 1:
    patient_rows[0][0] = str(patient_rows[0][0])
    filtered_patient_data.append(patient_rows[0])
  else:
    for patient_row in patient_rows:
      if found_positive: continue
      if patient_row[-3] == 1:
        patient_row[0] = str(patient_row[0])
        filtered_patient_data.append(patient_row)
        found_positive = True

# Add death status
df = sql_query("SELECT DISTINCT * FROM dw_test.orcl_cichum_bendeces_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhredeces > '2020-01-01'")

dead = [row.dossier for index, row in df.iterrows()]
final_patient_data = []

for row in filtered_patient_data:
  final_row = row
  if row[0] in dead:
    final_row = final_row + [1]
  else:
    final_row = final_row + [2]
  final_patient_data.append(final_row)

write_csv(TABLE_COLUMNS['patient_data'], final_patient_data, 
  os.path.join(CSV_DIRECTORY, 'patient_data.csv'))