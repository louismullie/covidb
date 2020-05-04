#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import csv, re, os
import numpy as np
import pandas as pd

from constants import LOCAL_SITE_CODE, CSV_DIRECTORY, \
  TABLE_COLUMNS, LIVE_SHEET_FILENAME
from time_utils import get_datetime_seconds, \
  get_hours_between_datetimes
from identity_utils import generate_patient_uid, \
  generate_patient_site_uid, generate_accession_uid
from file_utils import write_csv, read_csv
from postgresql_utils import sql_query

from mappers import map_time, map_patient_ramq, \
  map_patient_covid_status, map_patient_age, map_patient_sex

live_sheet_rows = read_csv(LIVE_SHEET_FILENAME, remove_duplicates=True)

all_mrns = list(set(
  [str(row[0]) for row in live_sheet_rows]
))

all_tests_obj = [
  [str(row[0]), str(row[-7]), row[-3] == 'External'] \
   for row in live_sheet_rows]

all_tests = [','.join([str(x) for x in y]) \
  for y in all_tests_obj]

total_tests = len(all_tests)

non_er_episodes_df = sql_query(
  "SELECT dossier, dhreadm, dhredep, unitesoinscode FROM " +
  "dw_test.cichum_sejhosp_live WHERE " +
  "dossier in (" + ", ".join(all_mrns) + ") " +
  "AND dhreadm > '2020-01-01'"
)

er_episodes_df = sql_query(
  "SELECT dossier, dhreadm, dhredep FROM " +
  "dw_test.orcl_cichum_sejurg_live WHERE " +
  "dossier in ('" + "', '".join(all_mrns) + "') " +
  "AND dhreadm > '2020-01-01'"
)

non_er_episodes = [
  [str(row.dossier), str(row.dhreadm), str(row.dhredep)] 
  for i, row in non_er_episodes_df.iterrows()
]

er_episodes = [
  [str(row.dossier), str(row.dhreadm), str(row.dhredep)] 
  for i, row in er_episodes_df.iterrows()
]

def is_between_datetimes(dt, d1, d2):
  delta_low = get_hours_between_datetimes(d1, dt)
  delta_high = get_hours_between_datetimes(dt, d2, default_now=True)
  return delta_low > 0 and delta_high > 0

test_num = 0
tests_done_external = []
tests_done_internal = []
tests_done_in_er_episode = []
tests_done_in_non_er_episode = []
tests_done_without_episode = []

overlapping_episodes = 0

for test_obj in all_tests_obj:
  test_mrn, test_dt, is_external = test_obj
  test = ','.join([str(x) for x in test_obj])

  if is_external:
    tests_done_external.append(test)
  else:
    tests_done_internal.append(test)

  was_done_in_er = False
  curr_ep = None
  for ep_mrn, ep_start, ep_end in er_episodes:
    if test_mrn != ep_mrn: continue
    if is_between_datetimes(test_dt, ep_start, ep_end):
      was_done_in_er = True
      break

  was_done_in_non_er = False
  for ep_mrn, ep_start, ep_end in non_er_episodes:
    if test_mrn != ep_mrn: continue
    if is_between_datetimes(test_dt, ep_start, ep_end):
      was_done_in_non_er = True
      break

  if was_done_in_er:
    tests_done_in_er_episode.append(test)
  elif was_done_in_non_er:
    tests_done_in_non_er_episode.append(test)
  else:
    tests_done_without_episode.append(test)

  test_num += 1

print('\n\n===== 1. Summary of COVID testing according to patient contact\n\n')

print('Total tests in live sheet: %d' % len(set(all_tests)))

print('\n... Tests done at outside labs: %d' % len(set(tests_done_external)))
print('   ... sampled during presence in CHUM emergency room: %d' % len(set(tests_done_in_er_episode) & set(tests_done_external)))
print('   ... sampled during presence on CHUM inpatient unit*: %d' % len(set(tests_done_in_non_er_episode) & set(tests_done_external)))
print('   ... sampled during out-of-hospital clinical episode: %d' % len(set(tests_done_without_episode) & set(tests_done_external)))

print('\n... Tests done at the CHUM labs: %d' % len(set(tests_done_internal)))
print('   ... sampled during presence in CHUM emergency room: %d' % len(set(tests_done_in_er_episode) & set(tests_done_internal)))
print('   ... sampled during presence on CHUM inpatient unit*: %d' % len(set(tests_done_in_non_er_episode) & set(tests_done_internal)))
print('   ... sampled during out-of-hospital clinical episode: %d' % len(set(tests_done_without_episode) & set(tests_done_internal)))

print('\n   * i.e. inpatient ward, intensive care unit, etc.')

print('\n\n===== 2. Summary of hospital contact by tested patients\n\n')

non_external_mrns = [
  str(row[0]) for row in live_sheet_rows \
  if row[-3] != 'External'
]

external_mrns = [
  mrn for mrn in all_mrns \
  if mrn not in non_external_mrns
]

num_tested = len(all_mrns)
num_external = len(set(external_mrns))

# 2. Number of hospital admissions
df = sql_query(
  "SELECT dossier, dhreadm FROM " +
  "dw_test.cichum_sejhosp_live WHERE " +
  "dossier in (" + ", ".join(all_mrns) + ") " +
  "AND dhreadm > '2020-01-01'"
)

admitted_mrns = [str(row.dossier) for i, row in df.iterrows()]
num_admissions = len(admitted_mrns)
num_admitted = len(set(admitted_mrns))

# 3. Number of ER visits
df = sql_query(
  "SELECT dossier, dhreadm FROM " +
  "dw_test.orcl_cichum_sejurg_live WHERE " +
  "dossier in ('" + "', '".join(all_mrns) + "') " +
  "AND dhreadm > '2020-01-01'"
)

visited_mrns = [str(row.dossier) for i, row in df.iterrows()]
num_visits = len(visited_mrns)
num_visited = len(set(visited_mrns))

# 4. Number of patients with at least one episode
included_mrns = list(set(admitted_mrns + visited_mrns))
num_included = len(included_mrns)

# 5. Patients who bypassed ER
admitted_from_er_mrns = [
  mrn for mrn in admitted_mrns if \
  mrn in visited_mrns
]

bypassed_er_mrns = [
  mrn for mrn in admitted_mrns if \
  mrn not in visited_mrns
]

df = sql_query(
  "SELECT dossier, unitesoinscode FROM " +
  "dw_test.cichum_sejhosp_live WHERE " +
  "dossier in (" + ", ".join(bypassed_er_mrns) + ") " +
  "AND dhreadm > '2020-01-01'")

hotel_dieu_mrns = [
  str(row.dossier) for i,row in df.iterrows() \
  if ('HLR' in str(row.unitesoinscode) or \
      'HDB' in str(row.unitesoinscode))
]

day_surgery_mrns = [
  str(row.dossier) for i,row in df.iterrows() \
  if ('CJ' in str(row.unitesoinscode))
]

cath_lab_mrns = [
  str(row.dossier) for i,row in df.iterrows() \
  if ('CHEM' in str(row.unitesoinscode))
]

no_unit_mrns = [
  str(row.dossier) for i,row in df.iterrows() \
  if (row.unitesoinscode is None or \
    str(row.unitesoinscode).strip() == '')
]

other_bypassed_mrns = [
  str(row.dossier) for i,row in df.iterrows() \
  if (str(row.dossier) not in hotel_dieu_mrns and
      str(row.dossier) not in day_surgery_mrns and
      str(row.dossier) not in cath_lab_mrns and 
      str(row.dossier) not in no_unit_mrns)
]

# 6. Print cohort information
print('Total patients tested in live sheet: %d' % \
  (num_tested))

print('Total patients with hospital contact: %d' % \
  (num_included))

print('\nTotal %d CHUM ER episodes (%d patients)' % \
  (num_visits, num_visited))

print('\nTotal %d hospital admissions (%d patients)' % \
  (num_admissions, num_admitted))

print('\n... %d admissions from ER (%d patients)' % \
  (len(admitted_from_er_mrns), \
   len(list(set(admitted_from_er_mrns)))))

print('\n... %d admissions from outside (%d patients)' % \
  (len(other_bypassed_mrns), \
   len(list(set(other_bypassed_mrns)))))

print('   ... i.e. direct-to-ward transfer or elective admission')

print('\nTotal episodes in other categories: %d' % \
  (num_admissions - len(admitted_from_er_mrns) \
  - len(other_bypassed_mrns)))

print('   ... %d admissions at Hotel-Dieu (%d patients)' % \
  (len(hotel_dieu_mrns), \
   len(list(set(hotel_dieu_mrns)))))

print('   ... %d presence in day surgery (%d patients)' % \
  (len(day_surgery_mrns), \
   len(list(set(day_surgery_mrns)))))

print('   ... %d presence in cath lab (%d patients)' % \
  (len(cath_lab_mrns), \
   len(list(set(cath_lab_mrns)))))

print('   ... %d episodes missing unit information (%d patients)' % \
  (len(no_unit_mrns), \
   len(list(set(no_unit_mrns)))))

# Begin building database
patient_data = {}
patient_mrns = []
patient_covid_statuses = {}

for row in live_sheet_rows:

  patient_mrn = str(row[0])

  #if patient_mrn in other_bypassed_mrns:
  #  continue

  patient_ramq = str(row[1])
  patient_age = row[-1]
  patient_birth_sex = row[-2]
  patient_covid_status = map_patient_covid_status(row[-4])
  
  pcr_result_time = row[-6]
  pcr_sample_time = row[-7]

  if patient_mrn not in patient_covid_statuses:
    patient_covid_statuses[patient_mrn] = [patient_covid_status]
  else:
    patient_covid_statuses[patient_mrn].append(patient_covid_status)

  if patient_mrn not in patient_data:
    patient_data[patient_mrn] = []

  patient_data[patient_mrn].append([
    patient_mrn,
    map_patient_ramq(patient_ramq),
    map_time(pcr_sample_time),
    LOCAL_SITE_CODE,
    patient_covid_status,
    map_patient_age(patient_age),
    map_patient_sex(patient_birth_sex)
  ])

  patient_mrns.append(patient_mrn)

filtered_patient_data = []

status_col = TABLE_COLUMNS['patient_data'].index('patient_covid_status')

# Build a list of unique patients
for patient_mrn in patient_data:

  patient_rows = patient_data[patient_mrn]
  
  # Cohort entry time is the time of the first PCR result.
  patient_rows.sort(key=lambda x: get_datetime_seconds(x[2]))

  patient_status = 'unknown'
  
  row_index = 0
  for patient_row in patient_rows:
    if patient_row[status_col] == 'positive':
      patient_status = 'positive'
      break
    elif patient_row[status_col] == 'negative':
      patient_status = 'negative'
    elif patient_row[status_col] == 'pending' and \
      patient_status != 'negative':
      patient_status = 'pending'
    row_index += 1
 
  if patient_status == 'positive':
    patient_row = patient_rows[row_index]
  else:
    patient_row = patient_rows[0]

  patient_row[status_col] = patient_status

  filtered_patient_data.append(patient_row)

# Add vital status
df = sql_query("SELECT dossier FROM " + \
  "dw_test.orcl_cichum_bendeces_live WHERE " + \
  "dossier in ('" + "', '".join(patient_mrns) + "') " + \
  "AND dhredeces > '2020-01-01'")

dead = [row.dossier for index, row in df.iterrows()]
final_patient_data = []

for row in filtered_patient_data:
  final_row = row
  if row[0] in dead:
    final_row = final_row + ['dead']
  else:
    final_row = final_row + ['alive']
  if row[0] in admitted_mrns:
    final_row = final_row + ['yes']
  else:
    final_row = final_row + ['no']
  final_patient_data.append(final_row)

write_csv(TABLE_COLUMNS['patient_data'], final_patient_data, 
  os.path.join(CSV_DIRECTORY, 'patient_data.csv'))