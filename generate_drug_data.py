#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY, DRUG_SKIP_VALUES
from postgresql_utils import sql_query
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid
from mappers import map_time, map_drug_name, map_drug_route, map_drug_frequency

row_count = 0
patient_data_rows = []
patient_mrns = []
patient_covid_statuses = {}
pcr_sample_times = {}

patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

for row in patient_data_rows:
  patient_mrn = str(row[0])
  patient_mrns.append(patient_mrn)
  pcr_sample_times[patient_mrn] = row[2]

df = sql_query("SELECT dossier, longdesc, orderstartdtm, componentstrengthnm, " + \
  "orderstopdtm, routecd, intervalsig, routecd FROM dw_v01.oacis_rx WHERE " + \
  "orderstartdtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

drug_data_rows = []
a = []
for index, row in df.iterrows():

  if row.longdesc.lower() in DRUG_SKIP_VALUES:
    continue

  patient_mrn = row.dossier
  drug_name = map_drug_name(row.longdesc)
  a.append(drug_name)
  drug_start_time = row.orderstartdtm
  drug_end_time = row.orderstopdtm
  
  delta_hours = get_hours_between_datetimes(
    pcr_sample_times[str(patient_mrn)], str(drug_start_time))
  
  if delta_hours < -48: continue

  drug_data_rows.append([
    patient_mrn, 
    drug_name, 
    map_time(drug_start_time), 
    map_time(drug_end_time),
    map_drug_frequency(row.intervalsig), 
    map_drug_route(row.routecd)
  ])

print(np.unique(a))
#print(drug_data_rows)
print('Total rows: %d' % len(drug_data_rows))

write_csv(TABLE_COLUMNS['drug_data'], drug_data_rows, 
  os.path.join(CSV_DIRECTORY, 'drug_data.csv'))