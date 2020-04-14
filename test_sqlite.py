import os, csv, sqlite3
import pandas as pd

from constants import SQLITE_DIRECTORY, CSV_DIRECTORY
db_file_name = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')

# Select all COVID positive patients
conn = sqlite3.connect(db_file_name)
curr = conn.execute("SELECT * from patient_data where patient_covid_status=1 or patient_covid_status=3")
res = curr.fetchall()
print('Total patients: %d' % len(res))
patient_site_uid = str(res[0][0])

# Select a specific
conn = sqlite3.connect(db_file_name)
curr = conn.execute("SELECT * from patient_data where patient_site_uid='" + patient_site_uid + "'")
res = curr.fetchall()

curr = conn.execute("SELECT * from lab_data where lab_name = 'Lympho #' AND patient_site_uid='" + patient_site_uid + "'")
res = curr.fetchall()
print(res)

# Fetch all COVID PCR tests for one patient
curr = conn.execute("SELECT * from pcr_data where pcr_name LIKE '%COVID%' AND patient_site_uid='" + patient_site_uid + "'")
res = curr.fetchall()
print(res)

curr = conn.execute("SELECT * from micro_data where micro_result_value='Pos'")
res = curr.fetchall()
print(len(res))

# Fetch all imaging tests
curr = conn.execute("SELECT * from imaging_data where imaging_modality='XR' AND imaging_site='chest'")
res = curr.fetchall()
print(len(res))

# Fetch all imaging tests
curr = conn.execute("SELECT * from imaging_data where patient_site_uid='" + patient_site_uid + "'")
res = curr.fetchall()
print(res)