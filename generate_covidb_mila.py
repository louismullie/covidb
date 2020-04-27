import subprocess, os
import pandas as pd
import numpy as np

from postgresql_utils import sql_query
from file_utils import read_csv, write_csv
from time_utils import get_hours_between_datetimes
from constants import CSV_DIRECTORY, TABLE_COLUMNS
from cli_utils import tabulate_columns

MILA_CSV_DIRECTORY = '/data8/projets/Mila_covid19/output/covidb_mila/csv'

patient_data_rows = read_csv(
  os.path.join(CSV_DIRECTORY, 'patient_data.csv'))

patient_mrns = []
pcr_sample_times = {}

for row in patient_data_rows:
  patient_mrn = row[0]
  patient_mrns.append(patient_mrn)
  pcr_sample_times[patient_mrn] = row[2]

df = sql_query("SELECT * FROM dw_v01.dw_rad_examen "+
  "WHERE dossier IN ('S" + "', 'S".join(patient_mrns) + "') " +
  "AND date_heure_exam > '2020-01-01'")

imaged_patient_mrns = []

for index, row in df.iterrows():
  lower_desc = row.description.lower()
  row_patient_mrn = str(row.dossier)[1:]

  if ('rx' in lower_desc and 'poumon' in lower_desc):
    hours = get_hours_between_datetimes(
      pcr_sample_times[row_patient_mrn], row.date_heure_exam)
    
    if hours < -48: continue
    imaged_patient_mrns.append(row_patient_mrn)

imaged_patient_mrns = np.unique(imaged_patient_mrns)

print("Number of patients with imaging: %d" % len(imaged_patient_mrns))

for table_name in TABLE_COLUMNS:
  
  if table_name == 'lab_data': continue

  table_file_name = os.path.join(CSV_DIRECTORY, table_name + '.csv')
  csv_rows = read_csv(table_file_name)
  column_names = TABLE_COLUMNS[table_name]
  filtered_csv_rows = [row for row in csv_rows if row[0] in imaged_patient_mrns]

  column_index = 0
  excluded_values_by_column = {}

  for column_name in column_names:
    excluded_values_by_column[column_name] = []

    column = np.asarray(
      [row[column_index] for row in filtered_csv_rows]
    )

    unique_values = np.unique(column)
    
    if len(unique_values) < 100: # to improve
      for unique_value in unique_values:
        count = np.count_nonzero(column == unique_value)
        if count < 5:
          #print(unique_value)
          #print('LOW ELEMENT COUNT: ' + str(count))
          excluded_values_by_column[column_name].append(unique_value)

    column_index += 1
  
  censored_csv_rows = []
  for filtered_csv_row in filtered_csv_rows:
    censored_csv_row = []
    column_index = 0
    for column_name in column_names:
      item = filtered_csv_row[column_index]
      #if item in excluded_values_by_column[column_name]:
      #  item = 'XX'
      censored_csv_row.append(item)
      column_index += 1
    censored_csv_rows.append(censored_csv_row)
  
  print('\n\n' + table_name + '\n\n')
  tabulate_columns(column_names[0:5], censored_csv_rows)

  if len(column_names) > 5:
    tabulate_columns(column_names[5:], censored_csv_rows, offset=5)

  filtered_table_file_name = os.path.join(MILA_CSV_DIRECTORY, table_name + '.csv')
  #print(censored_csv_rows)
  write_csv(column_names, censored_csv_rows, filtered_table_file_name)
