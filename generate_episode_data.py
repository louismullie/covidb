#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY, LIVE_SHEET_FILENAME
from postgresql_utils import sql_query, list_columns, list_tables
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_site_uid
from mappers import map_time, map_string_lower, map_episode_unit_type

row_count = 0
patient_data_rows = []
patient_mrns = []
patient_covid_statuses = {}
pcr_sample_times = {}

# Get MRNs of patients in cohort
patient_data_rows = read_csv(os.path.join( \
  CSV_DIRECTORY, 'patient_data.csv'))

patient_mrns = []

live_sheet_rows = read_csv(LIVE_SHEET_FILENAME, remove_duplicates=True)

pcr_sample_times = {}

for row in live_sheet_rows:
  patient_mrn = str(row[0])
  patient_mrns.append(patient_mrn)
  pcr_time = str(row[-7])
  if patient_mrn not in pcr_sample_times:
    pcr_sample_times[patient_mrn] = []
  pcr_sample_times[patient_mrn].append(pcr_time)

# Get ER visit data from ADT
df = sql_query(
  "SELECT dossier, noadm, dhreadm, dhredep, diagdesc FROM " +
  "dw_test.orcl_cichum_sejurg_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhreadm > '2020-01-01'"
)

episode_data_rows = [
  [
    map_string_lower(row.dossier), 
    map_string_lower(int(row.noadm)), 
    map_episode_unit_type('ER', None),  
    map_time(row.dhreadm), 
    map_time(row.dhredep),
    map_string_lower(row.diagdesc),
    get_hours_between_datetimes( \
      row.dhreadm, row.dhredep, default_now=True),
  ] for i, row in df.iterrows() \
  if row.dhreadm != '' and \
    pcr_sample_times[str(row.dossier)][0] != '' and \
    get_hours_between_datetimes( \
    row.dhreadm, pcr_sample_times[str(row.dossier)][0]
  ) > 0
]

episode_ids = [
  [map_string_lower(row.dossier)]
  for i, row in df.iterrows()
]

# Get inpatient location data from ADT
# 
df = sql_query(
  "SELECT  dossier, noadm, dhredeb, dhrefin, " +
  "localno, unitesoinscode, raison " +
  "FROM dw_test.ci_sejhosp_lit_live WHERE " +
  "dossier in (" + ", ".join(patient_mrns) + ") AND " +
  "dhredeb > '2020-01-01' ORDER BY dhredeb ASC"
)

locations_data = {} 
episode_ids = []

for index, row in df.iterrows():
  
  patient_mrn = str(row.dossier)
  episode_id = str(int(row.noadm))
  episode_ids.append(episode_id)

  location_start_time = map_time(row.dhredeb)

  # Unfortunately, not filled out in database
  # Leave this for later just in case
  # location_end_time = map_time(row.dhrefin)
  
  # Unfortunately, not filled out in database
  # Leave this for later just in case
  # location_description = str(row.raison).lower().strip()

  location_ward_code = str(row.unitesoinscode).strip()
  
  patient_location_data = {
    'location_ward_code': location_ward_code, 
    'location_start_time': location_start_time,
    'location_end_time': '',
    'location_description': ''
  }
  if patient_mrn == '5632640':
    print(episode_id, location_start_time, location_ward_code)

  if patient_mrn not in locations_data:
    locations_data[patient_mrn] = {}

  if episode_id not in locations_data[patient_mrn]:
    locations_data[patient_mrn][episode_id] = []

  current_num_locations = len(locations_data[patient_mrn][episode_id])
   
  # Skip patients going through Unite fantome
  if location_ward_code in ['CEC1', 'CEC2', 'CELJ','CBI']: continue

  # Handle changes between units of the same type as one location
  if current_num_locations > 1:
    previous_location = locations_data[patient_mrn][episode_id][-1]
    previous_ward_code = previous_location['location_ward_code']
    previous_ward_type = map_episode_unit_type( \
      previous_ward_code, location_start_time)
    current_ward_type = map_episode_unit_type( \
      location_ward_code, location_start_time)
    if previous_ward_type == current_ward_type:
      continue

  locations_data[patient_mrn][episode_id] \
    .append(patient_location_data)

episode_ids = np.unique(episode_ids)

# Get admissions data from ADT
df = sql_query(
  "SELECT dossier, noadm, dhreadm, dhredep, diagdesc " +
  " FROM dw_test.cichum_sejhosp_live WHERE " +
  " noadm in (" + ", ".join(episode_ids) + ")"
)

admissions_data = {}

for index, row in df.iterrows():
  
  patient_mrn = str(row.dossier)
  episode_id = str(int(row.noadm))

  admission_start_time = map_time(row.dhreadm)
  admission_end_time = map_time(row.dhredep)

  if row.diagdesc is None:
    admission_description = ''
  else:
    admission_description = str(row.diagdesc).lower().strip()
  
  admission_data = {
    'admission_start_time': admission_start_time,
    'admission_end_time': admission_end_time,
    'admission_description': admission_description
  }

  admissions_data[episode_id] = admission_data

# Piece together location and admissions data
for patient_mrn in locations_data:
  patient_locations_data = locations_data[patient_mrn]

  for episode_id in patient_locations_data:
    
    if episode_id not in admissions_data:
      admission_data = None
    else:
      admission_data = admissions_data[episode_id]

    episode_locations_data = patient_locations_data[episode_id]
    episode_location_total = len(episode_locations_data)

    for episode_location_num in range(0, episode_location_total):
      
      current_location = episode_locations_data[episode_location_num]
      location_ward_code = current_location['location_ward_code']
      
      episode_start_time = current_location['location_start_time']
      episode_unit_type = map_episode_unit_type( \
        location_ward_code, location_start_time)
      
      if episode_location_num + 1 < episode_location_total:
        next_location = episode_locations_data[episode_location_num + 1]
        admission_start_time = episode_start_time
        episode_end_time = next_location['location_start_time']
      elif admission_data is not None:
        admission_start_time = admission_data['admission_start_time']
        admission_end_time = admission_data['admission_end_time']
        if admission_end_time:
          episode_end_time = admission_end_time
        else:
          episode_end_time = ''
      else:
        episode_end_time = ''

      if admission_data is None:
        episode_description = current_location['location_description']
      else:
        episode_description = admission_data['admission_description']
      
      # Get length of stay
      episode_duration_hours = int(get_hours_between_datetimes(
        episode_start_time, episode_end_time, default_now=True
      ))
      # Skip if ADMISSION started > 5 days after first PCR test
      found_close = False
      for pcr_time in pcr_sample_times[patient_mrn]:

        hours_delta = get_hours_between_datetimes(
          admission_start_time, pcr_time)

        if hours_delta > -24*7 and hours_delta < 24*30: 
          found_close = True
      
      if not found_close: continue
      
      episode_data_rows.append([
        patient_mrn, 
        episode_id,
        episode_unit_type,
        episode_start_time, 
        episode_end_time,
        episode_description,
        episode_duration_hours
      ])

print('Total rows: %d' % len(episode_data_rows))

write_csv(TABLE_COLUMNS['episode_data'], episode_data_rows, 
  os.path.join(CSV_DIRECTORY, 'episode_data.csv'))