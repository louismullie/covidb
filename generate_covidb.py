import subprocess, os
import pandas as pd
import numpy as np

from constants import CODE_DIRECTORY, CSV_DIRECTORY

def most_frequent(data):
    return max(set(data), key=data.count)

def run_step(step_name):

  process_file_name = 'generate_' + step_name + '.py'

  subprocess.call([
  'python '+os.path.join(CODE_DIRECTORY, process_file_name)
  ], shell=True)

  csv_path = os.path.join(CSV_DIRECTORY, step_name + '.csv')

  print('\n\n* Table summary: %s' % step_name)

  df = pd.read_csv(csv_path)
  #df.describe()
  col_i = 0

  print(' \___ no. total rows: %d' % (df.size / len(df.columns)))

  for col in df.columns:
    col_data = []
    for row_j, row in df.iterrows():
      col_data.append(row[col_i])
    col_i += 1

    null_col_data = [1 if x == None else 0 for x in col_data]
    empty_col_data = [1 if str(x) == '' else 0 for x in col_data]

    print('\n  + ' + col)
    print('    \___ no. null values: %d' % np.count_nonzero(null_col_data))
    print('    \___ no. empty values: %d' % np.count_nonzero(empty_col_data))

    # Sample the first 50 to avoid computationally intensive calculations
    uniq_col_data = np.unique(col_data[0:50])

    if (float(len(uniq_col_data)) / len(col_data)) < 0.75:

      most_freq = most_frequent(col_data)
      uniq_col_data = np.unique(col_data)
      pct_uniq = round(float(len(uniq_col_data)) / len(col_data) * 100, 2)
      mf_col_data = [1 if x == most_freq else 0 for x in col_data]
      pct_most_freq = round(float(np.count_nonzero(mf_col_data)) / len(col_data) * 100, 2)

      print('    \___ no. unique values: %d' % len(uniq_col_data))

      if len(uniq_col_data) > 1 and pct_uniq > 10 and pct_uniq < 75:
        print('    \___ pct. of values unique: %f' % round(pct_uniq, 2))
        print('    \___ most frequent value: %s' % most_freq)
        print('    \___ pct. with most frequent value: %s' % pct_most_freq)

  print('\n\nWrote table to: %s' % csv_path)

print('\n\n* Generating patient data...')
run_step('patient_data')

print('\n\n* Generating episode data...')
run_step('episode_data')

print('\n\n* Generating diagnosis data...')
run_step('episode_data')

print('\n\n* Generating lab data...')
run_step('lab_data')

print('\n\n* Generating PCR data...')
run_step('pcr_data')

print('\n\n* Generating micro data...')
run_step('micro_data')

print('\n\n* Generating slice data...')
run_step('slice_data')

print('\n\n* Generating SQLite data...')
subprocess.call([
  'python '+os.path.join(CODE_DIRECTORY, 'generate_sqlite.py')
], shell=True)

print('\n\n* Done!')