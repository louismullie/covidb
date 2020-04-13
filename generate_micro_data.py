#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY
from sql_utils import sql_query, list_columns, list_tables
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid
from mappers import map_micro_sample_site

row_count = 0
patient_data_rows = []
patient_mrns = []
patient_covid_statuses = {}
pcr_sample_times = {}

patient_data_rows = read_csv(os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

for row in patient_data_rows:
  patient_mrn = str(row[0])
  patient_mrns.append(patient_mrn)
  patient_covid_statuses[patient_mrn] = row[-3]
  pcr_sample_times[patient_mrn] = row[1]

pcr_patterns = ['Culture', 'Cult', 'cult']

df = sql_query("SELECT * FROM dw_v01.oacis_mi WHERE longdesc NOT LIKE '%RIA%' AND (" +
    ' OR '.join(["longdesc LIKE '%" + pat + "%'" for pat in pcr_patterns]) + ") AND "
    "specimencollectiondtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

micro_data_rows = []

for index, row in df.iterrows():

  patient_mrn = str(row.dossier)
  micro_name = row.longdesc
  micro_sample_site = row.specimencollectionmethodcd
  micro_sample_time = row.specimencollectiondtm
  micro_result_time = row.resultdtm
  micro_result_value = row.growthcd

  micro_data_rows.append([
    patient_mrn, micro_name, 
    map_micro_sample_site(micro_sample_site),
    micro_sample_time, 
    micro_result_time, 
    micro_result_value, 
    patient_covid_statuses[patient_mrn]
  ])

print('Total rows: %d' % len(micro_data_rows))

write_csv(TABLE_COLUMNS['micro_data'], micro_data_rows, 
  os.path.join(CSV_DIRECTORY, 'micro_data.csv'))