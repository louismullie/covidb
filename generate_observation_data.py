#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv, os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY, \
 LAB_PENDING_FLAGS, LAB_CANCELLED_FLAGS, SIURG_OBSERVATION_NAMES
from postgresql_utils import sql_query
from file_utils import read_csv, write_csv
from time_utils import get_hours_between_datetimes
from mappers import map_time, map_float_value, map_observation_name

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
  observation_time = row.specimencollectiondtm
  observation_value = row.lbres_ck

  if observation_value is None or \
     observation_value in LAB_PENDING_FLAGS or \
     observation_value in LAB_CANCELLED_FLAGS:
    continue

  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[str(patient_mrn)], str(observation_time))
  
  if delta_hours > -48:
    observation_data_rows.append([
      patient_mrn, 
      map_observation_name(row.longdesc),
      map_time(observation_time),
      map_float_value(observation_value)
    ])

#### Add diagnoses from ER visits
df = sql_query("SELECT dossier, noadm, dhreadm FROM " + \
  "dw_test.orcl_cichum_sejurg_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhreadm > '2020-01-01'")

urg_episode_ids = []
episodes_by_id = {}

for i, row in df.iterrows():
  patient_mrn = str(row.dossier)
  episode_id = str(int(row.noadm))
 
  episode_start_time = str(row.dhreadm)
  episode = {
    'patient_mrn': patient_mrn,
    'episode_id': episode_id,
    'episode_start_time': episode_start_time
  }

  urg_episode_ids.append(episode_id)
  episodes_by_id[episode_id] = episode

df = sql_query("SELECT * from public.urgchum_episod_sv WHERE " + \
  "no_episod_is IN (" + ", ".join(urg_episode_ids) + ")")

for index, row in df.iterrows():
  
  episode_id = str(int(row.no_episod_is))
  episode = episodes_by_id[episode_id]
  patient_mrn = episode['patient_mrn']

  observation_time = map_time(row.dhm_sv)
  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[patient_mrn], observation_time)
  
  if delta_hours < -48: continue

  for observation_name in SIURG_OBSERVATION_NAMES:
    if observation_name not in row: continue
    observation_value = row.get(observation_name)
    if str(observation_value) == 'nan':
      continue
    observation_data_rows.append([
      patient_mrn, 
      map_observation_name(observation_name), 
      map_time(observation_time),
      map_float_value(observation_value)
    ])

#df = sql_query("SELECT * FROM dw_v01.oacis_ob WHERE " +
#    "rsltvalue IS NOT NULL AND " +
#    "dossier in (" + ", ".join(patient_mrns) + ") LIMIT 100")

# for index, row in df.iterrows():
  
#  print(row)
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

print('Total rows: %d' % len(observation_data_rows))

write_csv(TABLE_COLUMNS['observation_data'], observation_data_rows, 
  os.path.join(CSV_DIRECTORY, 'observation_data.csv'))