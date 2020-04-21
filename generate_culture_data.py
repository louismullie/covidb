#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, CSV_DIRECTORY
from postgresql_utils import sql_query
from file_utils import write_csv, read_csv
from time_utils import get_hours_between_datetimes
from identity_utils import generate_patient_uid, generate_patient_site_uid
from mappers import map_culture_sample_site, map_culture_growth_value

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
  pcr_sample_times[patient_mrn] = row[2]

pcr_patterns = ['Culture', 'Cult', 'cult']

df = sql_query("SELECT * FROM dw_v01.oacis_mi WHERE longdesc NOT LIKE '%RIA%' AND (" +
    ' OR '.join(["longdesc LIKE '%" + pat + "%'" for pat in pcr_patterns]) + ") AND "
    "specimencollectiondtm > '2020-01-01' AND dossier in (" + ", ".join(patient_mrns) + ")")

culture_data_rows = []

for index, row in df.iterrows():

  patient_mrn = str(row.dossier)
  culture_name = row.longdesc
  culture_sample_site = row.specimencollectionmethodcd
  culture_sample_time = row.specimencollectiondtm
  culture_result_time = row.resultdtm
  culture_growth_value = row.growthcd

  culture_data_rows.append([
    patient_mrn, culture_name, 
    map_culture_sample_site(culture_sample_site),
    culture_sample_time, 
    culture_result_time, 
    map_culture_growth_value(culture_growth_value)
  ])

print('Total rows: %d' % len(culture_data_rows))

write_csv(TABLE_COLUMNS['culture_data'], culture_data_rows, 
  os.path.join(CSV_DIRECTORY, 'culture_data.csv'))