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

# 1. Number of patients tested
all_mrns = [str(row[0]) for row in live_sheet_rows]
external_mrns = [str(row[0]) for row in live_sheet_rows if row[-3] == 'External']
num_tests = len(all_mrns)
num_tested = len(set(all_mrns))
num_external = len(set(external_mrns))

# 2. Number of hospital admissions
df = sql_query("SELECT dossier FROM dw_test.ci_sejhosp_lit_live WHERE " + \
  "dossier in (" + ", ".join(all_mrns) + ") " + \
  "AND dhredeb > '2020-01-01'")

admitted_mrns = [str(row.dossier) for i, row in df.iterrows()]
num_admissions = len(admitted_mrns)
num_admitted = len(set(admitted_mrns))

# 3. Number of ER visits
df = sql_query("SELECT dossier FROM dw_test.orcl_cichum_sejurg_live WHERE " + \
  "dossier in ('" + "', '".join(all_mrns) + "') " + \
  "AND dhreadm > '2020-01-01'")

visited_mrns = [str(row.dossier) for i, row in df.iterrows()]
num_visits = len(visited_mrns)
num_visited = len(set(visited_mrns))

# 4. Number of non-external patients
included_mrns = list(set(admitted_mrns + visited_mrns))
num_included = len(included_mrns)

# 5. Print cohort information
print('\nTotal %d tests done (%d patients)' % \
  (num_tests, num_tested))

print('... %d patients flagged as external' % \
  num_external)

print('\nTotal %d ER visits (%d patients)' % \
  (num_visits, num_visited))

print('... %d patients were admitted' % \
  len(list(set(visited_mrns) & set(admitted_mrns))))

print('\nTotal %d admissions (%d patients)' % \
  (num_admissions, num_admitted))

print('\nTotal %d patients with hospital contact' % \
  (num_included))

print('... %d had been flagged as external' % \
  len(list(set(included_mrns) & set(external_mrns))))

# Begin building database

patient_data = {}
patient_mrns = []
patient_covid_statuses = {}

for row in live_sheet_rows:

  patient_mrn = str(row[0])

  if patient_mrn not in included_mrns:
    continue

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
      str(patient_mrn),
      map_patient_ramq(patient_ramq),
      pcr_sample_time,
      'chum',
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

status_col = TABLE_COLUMNS['patient_data'].index('patient_covid_status')

# Build a list of unique patients
for patient_mrn in patient_data:

  patient_rows = patient_data[str(patient_mrn)]
  patient_rows.sort(key=lambda x: get_datetime_seconds(x[2]))
  found_positive = False
  
  # Choose the first PCR entry - to be revised
  if 'positive' not in patient_covid_statuses[patient_mrn]:
    filtered_patient_data.append(patient_rows[0])
  else:
    for patient_row in patient_rows:
      if found_positive: continue
      if patient_row[status_col] == 'positive':
        filtered_patient_data.append(patient_row)
        found_positive = True

# Add vital status
df = sql_query("SELECT DISTINCT * FROM dw_test.orcl_cichum_bendeces_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhredeces > '2020-01-01'")

dead = [row.dossier for index, row in df.iterrows()]
final_patient_data = []

for row in filtered_patient_data:
  final_row = row
  if row[0] in dead:
    final_row = final_row + ['dead']
  else:
    final_row = final_row + ['alive']
  if row[0] in admitted_mrns:
    final_row = final_row + ['yes']
  else:
    final_row = final_row + ['no']
  final_patient_data.append(final_row)

write_csv(TABLE_COLUMNS['patient_data'], final_patient_data, 
  os.path.join(CSV_DIRECTORY, 'patient_data.csv'))