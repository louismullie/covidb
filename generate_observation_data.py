#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv, os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY
from postgresql_utils import sql_query
from file_utils import read_csv, write_csv
from time_utils import get_hours_between_datetimes
from mappers import map_observation_name

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

# extCOVID19
# SARS-CoV-2 Hors-CHUM
df = sql_query("SELECT * FROM dw_v01.oacis_lb WHERE " +
    "resultdtm IS NOT NULL AND longdesc in ('FIO2', 'Sat O2 Art', 'Température') AND " +
    "specimencollectiondtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

observation_data_rows = []

for index, row in df.iterrows():
  
  patient_mrn = str(row.dossier)
  observation_name = map_observation_name(row.longdesc)
  observation_time = row.specimencollectiondtm
  observation_value = row.lbres_ck

  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[str(patient_mrn)], str(observation_time))
  
  if delta_hours > -48:
    observation_data_rows.append([
      patient_mrn, observation_name, observation_time, observation_value
    ])

#df = sql_query("SELECT * FROM dw_v01.oacis_ob WHERE " +
#    "rsltvalue IS NOT NULL AND " +
#    "dossier in (" + ", ".join(patient_mrns) + ") LIMIT 100")


#for index, row in df.iterrows():
  
#  patient_mrn = str(row.dossier)
#  measurement_name = row.longdesc
#  #measurement_time = row.entereddtm
#  measurement_value = row.rsltvalue
#  measurement_units = row.unitcd

#  delta_hours = get_hours_between_datetimes(
#    pcr_sample_times[str(patient_mrn)], str(measurement_time))
  
#  if delta_hours > -48 and delta_hours < 7*24:
#    measurement_data_rows.append([
#      patient_mrn, measurement_name, measurement_time, measurement_value
#    ])

#df = sql_query("SELECT * FROM dw_v01.urg_sv_stack LIMIT 100 ") #WHERE " +
    #"dossier in (" + ", ".join(patient_mrns) + ") LIMIT 100")

#df = sql_query("SELECT * FROM dw_v01.urg_usager_stack LIMIT 100 ")
#print(df.iloc[2])
print('Total rows: %d' % len(observation_data_rows))

write_csv(TABLE_COLUMNS['observation_data'], observation_data_rows, 
  os.path.join(CSV_DIRECTORY, 'observation_data.csv'))