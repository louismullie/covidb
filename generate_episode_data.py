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
from mappers import map_time, map_episode_unit_type

row_count = 0
patient_data_rows = []
patient_mrns = []
patient_covid_statuses = {}
pcr_sample_times = {}

patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

for row in patient_data_rows:
  patient_mrn = str(row[0])
  patient_mrns.append(patient_mrn)

# Get ER episodes
df = sql_query("SELECT DISTINCT * FROM dw_test.orcl_cichum_sejurg_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhreadm > '2020-01-01'")

episode_data_rows = []

# Add episode visits
for index, row in df.iterrows():
  patient_mrn = str(row.dossier)
  episode_id = str(int(row.noadm))
  episode_unit_type = map_episode_unit_type('ER')
  episode_start_time = map_time(row.dhreadm)
  episode_end_time = map_time(row.dhredep)
  service_code = row.demadmservcode

  #print([row.typedestcode, episode_description])

  if row.diagdesc is None:
    episode_description = ''
  else:
    episode_description = str(row.diagdesc).lower().strip()

  episode_data_rows.append([
    patient_mrn, episode_id, episode_unit_type, 
    episode_start_time, episode_end_time,
    episode_description
  ])

# Get admission episodes
df = sql_query("SELECT DISTINCT * FROM dw_test.ci_sejhosp_lit_live WHERE " + \
  "dossier in (" + ", ".join(patient_mrns) + ") " + \
  "AND dhredeb > '2020-01-01'")

# Add admission episodes
for index, row in df.iterrows():
  patient_mrn = str(row.dossier)

  episode_id = str(int(row.noadm))
  episode_start_time = str(row.dhredeb)
  
  if row.dhrefin is None:
    episode_end_time = ''
  else:
    episode_end_time = str(row.dhrefin)

  room_number_str = str(row.localno)
  ward_code_str = str(row.unitesoinscode)
  episode_description = None

  if row.raison is None:
    episode_description = ''
  else:
    episode_description = str(row.raison).lower().strip()
  
  episode_data_rows.append([
    patient_mrn, 
    episode_id,
    map_episode_unit_type(ward_code_str),
    episode_start_time, 
    episode_end_time,
    episode_description
  ])

print('Total rows: %d' % len(episode_data_rows))

write_csv(TABLE_COLUMNS['episode_data'], episode_data_rows, 
  os.path.join(CSV_DIRECTORY, 'episode_data.csv'))