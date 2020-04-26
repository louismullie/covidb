#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import CSV_DIRECTORY, TABLE_COLUMNS, IMAGING_LIST_FILENAME
from postgresql_utils import sql_query
from file_utils import read_csv, write_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_accession_uid

patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

patient_mrns = []
pcr_sample_times = {}

for row in patient_data_rows:
  patient_mrn = row[0]
  patient_mrns.append(patient_mrn)
  pcr_sample_times[patient_mrn] = row[2]

df = sql_query("SELECT * FROM dw_v01.dw_rad_examen "+
  "WHERE dossier IN ('S" + "', 'S".join(patient_mrns) + "') " +
  "AND date_heure_exam > '2020-01-01'")

imaging_data_rows = []
patients_with_imaging = []
imaging_accession_numbers = []

for index, row in df.iterrows():
  lower_desc = row.description.lower()
  row_patient_mrn = str(row.dossier)[1:]

  if ('rx' in lower_desc and 'poumon' in lower_desc): #or \
     #('scan' in lower_desc and 'thorax' in lower_desc):
     #('scan' in lower_desc and 'abdo' in lower_desc):
    hours = get_hours_between_datetimes(
      pcr_sample_times[row_patient_mrn], row.date_heure_exam)
    
    if hours < -48: continue

    patients_with_imaging.append(row_patient_mrn)
    imaging_accession_numbers.append(row.accession_number)
      
    imaging_accession_uid = generate_accession_uid(row.accession_number)

    imaging_data_rows.append([
      row_patient_mrn,
      imaging_accession_uid,
      'xr', 'chest'
    ])

patients_with_imaging = np.unique(patients_with_imaging)
imaging_accession_numbers = np.unique(imaging_accession_numbers)

print('Number of patients with imaging: %d' % \
  len(patients_with_imaging))

write_csv(['accession_number'], \
 [[x] for x in imaging_accession_numbers], \
 IMAGING_LIST_FILENAME)

write_csv(TABLE_COLUMNS['imaging_data'], imaging_data_rows, 
  os.path.join(CSV_DIRECTORY, 'imaging_data.csv'))