#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv, os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY, \
 LAB_PENDING_FLAGS, LAB_CANCELLED_FLAGS, SIURG_OBSERVATION_NAMES, NP_TO_FIO2_MAP
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

observation_data_rows = []

df = sql_query(
  "SELECT dossier, servacro, rsltvalue, startdtm, unitcd FROM " +
  "dw_v01.oacis_ob WHERE " +
  "startdtm > '2020-01-01' AND " +
  "dossier in (" + ", ".join(patient_mrns) + ")"
)

#print('done')
#print(df.iloc[0])
#exit()

# Get vital signs from Oacis
#df = sql_query(
# "SELECT dossier, servtxt, startdtm, rsltvalue, unitcd FROM public.oacis_zrob INNER JOIN " +
# "public.oacis_zrobvalues ON oacis_zrobvalues.sid = oacis_zrob.sid INNER JOIN " +
# "datalake.oacis_serv ON oacis_serv.sid = oacis_zrob.sid INNER JOIN " +
# "dw_v01.dw_oacis_pat_doss ON oacis_serv.pid = dw_oacis_pat_doss.opid WHERE " +
# "startdtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")" 

#print(df.size)
for index, row in df.iterrows():

  patient_mrn = str(row.dossier)
  observation_name = row.servacro
  observation_value = row.rsltvalue
  observation_time = row.startdtm
  observation_unit = row.unitcd

  try:
    mapped_observation_name = map_observation_name(observation_name, observation_unit)
  except:
    continue

  mapped_observation_time = map_time(observation_time)
  mapped_observation_value = map_float_value(observation_value)

  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[str(patient_mrn)], str(observation_time))
  9
  if delta_hours < -48: continue

  observation_data_rows.append([
    patient_mrn, 
    mapped_observation_name,
    mapped_observation_time,
    mapped_observation_value
  ])

print('Done fetching observations from Oacis')

df = sql_query("SELECT * FROM dw_v01.oacis_lb WHERE " +
    "resultdtm IS NOT NULL AND longdesc in ('FIO2', 'Sat O2 Art', 'Température') AND " +
    "specimencollectiondtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

for index, row in df.iterrows():
  
  patient_mrn = str(row.dossier)
  observation_time = row.specimencollectiondtm
  observation_value = row.lbres_ck

  if observation_value is None or \
     str(observation_value) == 'nan' or \
     observation_value in LAB_PENDING_FLAGS or \
     observation_value in LAB_CANCELLED_FLAGS:
    continue
  
  mapped_observation_time = map_time(observation_time)
  mapped_observation_name = map_observation_name(row.longdesc, None)
  mapped_observation_value = map_float_value(observation_value)

  if mapped_observation_name == 'fraction_inspired_oxygen':
  
    # If FiO2 is encoded as 0, (relatively) safely assume 21%.
    # If FiO2 is encoded as 0.21, safely assume 21%.
    if float(mapped_observation_value) == 0 or \
       float(mapped_observation_value) == 0.21:
      mapped_observation_value = '21.0'
    # Cannot assume what this is, includes values incorrectly
    # encoded as fractions (e.g. 0.50 instead of 50%),
    # but also typos (e.g. 2.0 instead of 21.0) as well as 
    # nasal prongs flow rates incorrectly entered as FiO2s.
    elif float(mapped_observation_value) < 21.0:
      continue

  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[str(patient_mrn)], str(observation_time))
  
  if delta_hours > -48:

    observation_data_rows.append([
      patient_mrn, 
      mapped_observation_name,
      mapped_observation_time,
      mapped_observation_value
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

    observation_data_rows.append([
      patient_mrn, 
      map_observation_name(observation_name, None), 
      map_time(observation_time),
      map_float_value(observation_value)
    ])
 
  if row.sat_o2_cod is not None and row.sat_o2_cod != 'ND':
    oxygen_flow_rate, oxygenation_device, \
      fraction_inspired_oxygen = None, None, None
    if row.sat_o2_cod == 'AA':
      fraction_inspired_oxygen = '21.0'
      oxygenation_device = 'AA'
      oxygen_flow_rate = None
    elif row.sat_o2_cod == 'LN':
      oxygenation_device = 'LN'
      oxygen_flow_rate = str(row.sat_o2_qte_recue)
      if row.sat_o2_qte_recue > 10: 
        oxygen_flow_rate = '10.0'
      if oxygen_flow_rate == 'nan':
        oxygen_flow_rate = None
        fraction_inspired_oxygen = None
      else:
        fraction_inspired_oxygen = NP_TO_FIO2_MAP[oxygen_flow_rate]
    elif row.sat_o2_cod == 'VM':
      fraction_inspired_oxygen = str(row.sat_o2_qte_recue)
      oxygenation_device = 'VM'
      oxygen_flow_rate = None
    elif row.sat_o2_cod in ['VE', 'MA', 'BP']: # VE, MA, BP ?
      fraction_inspired_oxygen = str(row.sat_o2_qte_recue)
      oxygenation_device = None
      oxygen_flow_rate = None
    elif row.sat_o2_cod == 'OF':
      continue # tbd
    else:
      print(row)
      print('Invalid oxygenation parameters')
      exit()


    if fraction_inspired_oxygen is not None:

      if float(fraction_inspired_oxygen) < 1:
        print(fraction_inspired_oxygen)

      observation_data_rows.append([
        patient_mrn, 
        'fraction_inspired_oxygen', 
        map_time(observation_time),
        map_float_value(fraction_inspired_oxygen)
      ])

    if oxygen_flow_rate is not None:
      
      observation_data_rows.append([
        patient_mrn, 
        'oxygen_flow_rate', 
        map_time(observation_time),
        map_float_value(oxygen_flow_rate)
      ])

    #print(oxygenation_device, oxygen_flow_rate, fraction_inspired_oxygen)

#df2 = sql_query(
# "SELECT * from public.urgchum_episod_sn LIMIT 100 "
#)

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