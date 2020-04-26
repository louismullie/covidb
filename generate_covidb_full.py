import subprocess, os
import pandas as pd
import numpy as np

from constants import CODE_DIRECTORY, CSV_DIRECTORY

def run_step(step_name):

  process_file_name = 'generate_' + step_name + '.py'

  subprocess.call([
  'python '+os.path.join(CODE_DIRECTORY, process_file_name)
  ], shell=True)

  csv_path = os.path.join(CSV_DIRECTORY, step_name + '.csv')

  print('\n\n* Generated table: %s' % step_name)

  df = pd.read_csv(csv_path)
  #df.describe()
  col_i = 0

  print(' \___ no. total rows: %d' % (df.size / len(df.columns)))

  print('\n\nWrote table to: %s' % csv_path)

print('\n\n* Generating patient data...')
run_step('patient_data')

#print('\n\n* Generating imaging data...')
#run_step('imaging_data')

print('\n\n* Generating episode data...')
run_step('episode_data')

print('\n\n* Generating diagnosis data...')
run_step('diagnosis_data')

print('\n\n* Generating drug data...')
run_step('drug_data')

print('\n\n* Generating lab data...')
run_step('lab_data')

print('\n\n* Generating PCR data...')
run_step('pcr_data')

print('\n\n* Generating culture data...')
run_step('culture_data')

print('\n\n* Generating slice data...')
run_step('slice_data')

print('\n\n* Generating SQLite data...')
subprocess.call([
  'python '+os.path.join(CODE_DIRECTORY, 'generate_sqlite.py')
], shell=True)

print('\n\n* Done!')