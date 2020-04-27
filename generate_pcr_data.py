#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, LIVE_SHEET_FILENAME, CSV_DIRECTORY
from postgresql_utils import sql_query
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid
from mappers import map_time, map_pcr_result_value, map_pcr_result_status, \
  map_pcr_sample_site, map_pcr_name

row_count = 0
patient_data_rows = []
patient_mrns = []
patient_covid_statuses = {}
pcr_sample_times = {}

patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

for row in patient_data_rows:
  patient_mrn = str(row[0])
  patient_mrns.append(patient_mrn)
  patient_covid_statuses[patient_mrn] = row[-3]
  pcr_sample_times[patient_mrn] = row[2]

pcr_patterns = ['PCR', 'COVID', 'Influenza', 'RSV', 'Coronavirus']
sql_pcr_patterns = ["longdesc LIKE '%" + pat + "%'" for pat in pcr_patterns]

df = sql_query("SELECT * FROM dw_v01.oacis_lb WHERE (" +
    ' OR '.join(sql_pcr_patterns) + ") AND " + 
    "specimencollectiondtm > '2020-01-01' AND " +
    "dossier in (" + ", ".join(patient_mrns) + ")")

pcr_sample_times = {}
pcr_data_rows = []

for index, row in df.iterrows():

  patient_mrn = str(row.dossier)
  pcr_name = row.longdesc
  pcr_sample_site = row.specimencollectionmethodcd
  pcr_sample_time = row.specimencollectiondtm
  pcr_result_time = row.resultdtm
  pcr_result_value = row.lbres_ck
  pcr_result_units = row.resultunit

  if 'annul' in str(pcr_result_value) or \
    pcr_result_value is None or \
    map_pcr_result_value(pcr_result_value) == '':
    continue
  
  if patient_mrn not in pcr_sample_times:
    pcr_sample_times[patient_mrn] = []

  pcr_sample_times[patient_mrn].append(str(pcr_sample_time))
  
  pcr_data_rows.append([
    patient_mrn, 
    map_pcr_name(pcr_name), 
    map_pcr_sample_site(pcr_name, pcr_sample_site), 
    map_time(pcr_sample_time), 
    map_time(pcr_result_time), 
    map_pcr_result_value(pcr_result_value),
    map_pcr_result_status(pcr_result_value)
  ])

live_sheet_rows = read_csv(LIVE_SHEET_FILENAME, remove_duplicates=True)

for row in live_sheet_rows:

  is_external = (row[-3] == 'External')
  if is_external: continue

  patient_mrn = str(row[0])
  
  if patient_mrn in patient_mrns:

    pcr_result_time = row[-6]
    pcr_sample_time = row[-7]
    pcr_result_value = row[-4]

    if patient_mrn not in pcr_sample_times:
      pcr_sample_times[patient_mrn] = []

    if str(pcr_sample_time) in pcr_sample_times[patient_mrn]:
      #if DEBUG: print('Skipping duplicate PCRs')  
      continue
    else:
      pcr_sample_times[patient_mrn].append(str(pcr_sample_time))

    pcr_data_rows.append([
      patient_mrn, 
      map_pcr_name('covid-19 pcr'), 
      map_pcr_sample_site('covid-19 pcr', 'écouvillon nasal'), 
      map_time(pcr_sample_time), 
      map_time(pcr_result_time), 
      map_pcr_result_value(pcr_result_value),
      map_pcr_result_status(pcr_result_value)
    ])

print('Total rows: %d' % len(pcr_data_rows))

write_csv(TABLE_COLUMNS['pcr_data'], pcr_data_rows, 
  os.path.join(CSV_DIRECTORY, 'pcr_data.csv'), remove_duplicates=True)