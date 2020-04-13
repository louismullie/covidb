#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv, re, os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, LIVE_SHEET_FILENAME, CSV_DIRECTORY
from sql_utils import sql_query, list_columns
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes, get_datetime_seconds
from identity_utils import generate_patient_uid, generate_patient_site_uid
from mappers import map_patient_covid_status, map_patient_ramq, map_patient_age, map_patient_sex

live_sheet_rows = read_csv(LIVE_SHEET_FILENAME)

print('Number of rows in live sheet: %d' % len(live_sheet_rows))
patient_data = {}
patient_mrns = []

patient_ramqs = {}
pcr_sample_times = {}

for row in live_sheet_rows:

  is_external = (row[-3] == 'External')
  if is_external: continue

  patient_mrn = str(row[0])
  patient_mrn_string = 'S' + patient_mrn
  patient_ramq = row[1]

  if not validate_patient_ramq(patient_ramq):
    if DEBUG:
      print('Skipping patient with invalid RAMQ: %s' % patient_ramq)

  patient_age = row[-1]
  patient_birth_sex = row[-2]
  patient_covid_status = row[-4]
  
  pcr_result_time = row[-6]
  pcr_sample_time = row[-7]

  pcr_sample_times[patient_mrn_string] = pcr_sample_time
  patient_ramqs[patient_mrn_string] = patient_ramq

  try:

    if patient_mrn not in patient_data:
      patient_data[patient_mrn_string] = []

    patient_data[patient_mrn_string].append([
      patient_mrn,
      pcr_sample_time,
      generate_patient_uid(patient_ramq),
      generate_patient_site_uid(patient_mrn),
      'CHUM',
      '',
      map_patient_covid_status(patient_covid_status),
      map_patient_age(patient_age),
      map_patient_sex(patient_birth_sex)
    ])

  except Exception as err:

    if DEBUG:
      print('Skipping row: %s' % row)
      print('  due to error: %s' % err)
    continue

  patient_mrns.append(patient_mrn_string)

df = sql_query("SELECT * FROM dw_v01.dw_rad_examen "+
  "WHERE dossier IN ('" + "', '".join(patient_mrns) + "') " +
  "AND date_heure_exam > '2020-01-01'")

imaging_data_rows = []
patient_mrns_having_imaging = []
patient_ramqs_having_imaging = []
accession_numbers = []

for index, row in df.iterrows():
  lower_desc = row.description.lower()
  if ('rx' in lower_desc and 'poumon' in lower_desc): #or \
     #('scan' in lower_desc and 'thorax' in lower_desc):
     #('scan' in lower_desc and 'abdo' in lower_desc):
    hours = get_hours_between_datetimes(
      row.date_heure_exam, pcr_sample_times[row.dossier])
    
    if hours < 24:
      imaging_data_rows.append([
        row.dossier[1:-1],
        row.accession_number,
        'XR'
      ])
      patient_mrns_having_imaging.append(row.dossier)
      accession_numbers.append(row.accession_number)

write_csv(TABLE_COLUMNS['imaging_data'], imaging_data_rows, 
  os.path.join(CSV_DIRECTORY, 'imaging_data.csv'))

print('Number of patients with imaging: %d' % len( \
  np.unique(patient_mrns_having_imaging)))

patient_mrns_having_imaging = np.unique( \
  patient_mrns_having_imaging)

filtered_patient_data = []

for patient_mrn in patient_mrns_having_imaging:

  patient_rows = patient_data[patient_mrn]
  patient_rows.sort(key=lambda x: get_datetime_seconds(x[1]))
  found_positive = False
  for patient_row in patient_rows:
    if None in patient_row:
      print(patient_row)
    if patient_row[-3] == 1 or patient_row[-3] == 3:
      filtered_patient_data.append(patient_row)
      found_positive = True
  if not found_positive:
    filtered_patient_data.append(patient_rows[0])

write_csv(TABLE_COLUMNS['patient_data'], filtered_patient_data, 
  os.path.join(CSV_DIRECTORY, 'patient_data.csv'))