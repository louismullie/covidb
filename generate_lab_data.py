#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv, os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, LIVE_SHEET_FILENAME, \
  CSV_DIRECTORY, LAB_CANCELLED_FLAGS, LAB_SKIP_VALUES

from postgresql_utils import sql_query
from file_utils import read_csv, write_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid

from mappers import map_float_value, map_time, map_lab_name, \
  map_lab_sample_site, map_lab_result_value, map_observation_name

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
    pcr_sample_times[str(patient_mrn)] = row[2]
  row_count += 1

df = sql_query("SELECT * FROM dw_v01.oacis_lb WHERE " +
    "lbres_ck IS NOT NULL AND resultunit IS NOT NULL AND resultdtm IS NOT NULL AND " +
    "specimencollectiondtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

lab_data_rows = []

for index, row in df.iterrows():
  
  # Temperature and other observations encoded in labs
  if map_observation_name(row.longdesc) is not None:
    continue

  patient_mrn = str(row.dossier)
  lab_name = row.longdesc
 
  lab_sample_site = row.specimencollectionmethodcd
  lab_sample_time = row.specimencollectiondtm
  lab_result_time = row.resultdtm
  lab_result_string = row.lbres_ck
  lab_result_units = row.resultunit

  if lab_result_string in LAB_SKIP_VALUES:
    continue

  if lab_result_string in LAB_CANCELLED_FLAGS:
    lab_result_status = 'cancelled'
    lab_result_string = ''
  elif 'attente' in lab_result_string:
    lab_result_status = 'pending'
    lab_result_string = ''
  else:
    lab_result_status = 'resulted'

  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[str(patient_mrn)], str(lab_sample_time))
  
  if delta_hours > -48 and delta_hours < 7*24:
    lab_data_rows.append([
      patient_mrn, 
      map_lab_name(lab_name), 
      map_lab_sample_site(lab_sample_site), 
      map_time(lab_sample_time), 
      map_time(lab_result_time), 
      lab_result_status,
      lab_result_units, 
      lab_result_string, 
      map_lab_result_value(lab_result_string)
    ])

print('Total rows: %d' % len(lab_data_rows))

write_csv(TABLE_COLUMNS['lab_data'], lab_data_rows, 
  os.path.join(CSV_DIRECTORY, 'lab_data.csv'))