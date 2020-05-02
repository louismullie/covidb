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
from mappers import map_episode_unit_type, map_diagnosis_type

def pad_mrn(mrn):
  mrn_str = str(mrn)
  while len(mrn_str) < 10:
    mrn_str = '0' + mrn_str
  return mrn_str

def unpad_mrn(mrn):
  mrn_str = str(mrn)

  while mrn_str[0] == '0':
    mrn_str = mrn_str[1:]
  return mrn_str

## Get ICD codes
df = sql_query('SELECT * from dw_v01.icd10_codes')
icd10_codes = {}
for i, row in df.iterrows():
  category = row.categories[0:3]
  diagnosis = row.categories[3:]
  if diagnosis == '':
    diagnosis = row.subcat
  if diagnosis == '':
    code = category
  else:
    code = category + '.' + diagnosis
  icd10_codes[code] = row.desc3

df = sql_query('SELECT * from dw_v01.icd10_cat')
for i, row in df.iterrows():
  icd10_codes[row.categories] = row.description

icd10_codes['U07.1'] = 'COVID-19, virus identified'
icd10_codes['U07.2'] = 'COVID-19, virus not identified'
icd10_codes['I64'] = 'Stroke, not specified as hemorrhage or infarction'
icd10_codes['T13.0'] = 'Superficial injury of lower limb, level unspecified'
icd10_codes['T13.1'] = 'Open wound of lower limb, level unspecified'
icd10_codes['T35.7'] = 'Unspecified frostbite of unspecified site'

for code in icd10_codes:
  icd10_codes[code] = icd10_codes[code].lower()

patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))
patient_mrns = [str(row[0]) for row in patient_data_rows]
padded_patient_mrns = [pad_mrn(x) for x in patient_mrns]

episode_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'episode_data.csv'))

diagnosis_data_rows = []

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

df = sql_query("SELECT * from datalake.urgchum_diagno_episod WHERE " + \
  "no_episod_is IN (" + ", ".join(urg_episode_ids) + ")")

for index, row in df.iterrows():
  episode_id = str(int(row.no_episod_is))
  episode = episodes_by_id[episode_id]
  patient_mrn = episode['patient_mrn']
  diagnosis_icd_code = row.cod_gap
  if diagnosis_icd_code not in icd10_codes:
    cat = diagnosis_icd_code.split('.')[0]
    if cat in icd10_codes:
      diagnosis_name = icd10_codes[cat]
    else:
      print('Skipping: ' + diagnosis_icd_code)
      diagnosis_name = ''
  else:
    diagnosis_name = icd10_codes[diagnosis_icd_code]
  diagnosis_time = episode['episode_start_time']
  diagnosis_type = row.typ_diagno

  diagnosis_data_rows.append([
    patient_mrn, 
    episode_id, 
    map_diagnosis_type(diagnosis_type),
    diagnosis_name,
    diagnosis_icd_code, 
    diagnosis_time
  ])

### Add diagnoses from hospitalisations
df = sql_query("SELECT * FROM dw_test.ci_sejhosp_lit_live WHERE " + \
  "dossier in (" + ", ".join(patient_mrns) + ") " + \
  "AND dhredeb > '2020-01-01'")

hosp_episode_ids = []
episodes_by_id = {}
for i, row in df.iterrows():
  patient_mrn = str(row.dossier)
  episode_id = str(int(row.noadm))
 
  episode_start_time = str(row.dhredeb)
  episode = {
    'patient_mrn': patient_mrn,
    'episode_id': episode_id,
    'episode_start_time': episode_start_time
  }

  hosp_episode_ids.append(episode_id)
  episodes_by_id[episode_id] = episode

df = sql_query("SELECT * from dw_v01.mep_vue_diagnostic WHERE " + \
  "nodossier in ('" + "', '".join(padded_patient_mrns) + "') AND " + \
  "noepisode in ('" + "', '".join(hosp_episode_ids) + "') AND " + \
  "desctypediagnostic = 'Diagnostic principal'")

for index, row in df.iterrows():
  
  patient_mrn_padded = str(row.nodossier)
  patient_mrn = unpad_mrn(patient_mrn_padded)

  episode_id = str(int(row.noepisode))
  episode = episodes_by_id[episode_id]

  #diagnosis_name = row.descdiagnostic
  diagnosis_icd_code = row.cddiagnostic
  diagnosis_icd_code = diagnosis_icd_code \
   .replace('.1SV', '.1').replace('.1S', '.1') \
   .replace('.2S', '.2').replace('.2V', '.2')

  if diagnosis_icd_code not in icd10_codes:
    cat = diagnosis_icd_code.split('.')[0]
    if cat in icd10_codes:
      diagnosis_name = icd10_codes[cat]
    else:
      print('Skipping: ' + diagnosis_icd_code)
      diagnosis_name = ''
  else:
    diagnosis_name = icd10_codes[diagnosis_icd_code]
  diagnosis_time = episode['episode_start_time']

  diagnosis_data_rows.append([
    patient_mrn, 
    episode_id, 
    map_diagnosis_type(row.desctypediagnostic),
    diagnosis_name,
    diagnosis_icd_code, 
    diagnosis_time
  ])

### Add deaths
episode_ids = []
last_episode_by_dossier = {}
episodes_by_id = {}

# Find the last episode for each patient to attribute death
for row in episode_data_rows:
  
  patient_mrn = str(row[0])
  episode_id = str(row[1])
  episode_start_time = str(row[3])

  episode = {
    'episode_id': episode_id,
    'episode_start_time': episode_start_time
  }

  delta = 1
  if patient_mrn in last_episode_by_dossier:
    delta = get_hours_between_datetimes(
      last_episode_by_dossier[patient_mrn]['episode_start_time'],
      episode_start_time
    )
  
  if delta > 0:
    last_episode_by_dossier[patient_mrn] = episode

  episode_ids.append(episode_id)
  episodes_by_id[episode_id] = episode

df = sql_query("SELECT DISTINCT * FROM dw_test.orcl_cichum_bendeces_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhredeces > '2020-01-01'")

for index, row in df.iterrows():
  patient_mrn = str(row.dossier)
  
  last_episode_for_dossier = last_episode_by_dossier[patient_mrn]
  episode_id = last_episode_for_dossier['episode_id']
  diagnosis_type = 'death'
  diagnosis_time = row.dhredeces

  diagnosis_data_rows.append([
    patient_mrn, episode_id, diagnosis_type, 
    'death', '', diagnosis_time
  ])

print('Total rows: %d' % len(diagnosis_data_rows))

write_csv(TABLE_COLUMNS['diagnosis_data'], diagnosis_data_rows, 
  os.path.join(CSV_DIRECTORY, 'diagnosis_data.csv'))