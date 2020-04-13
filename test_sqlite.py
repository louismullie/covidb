import os, csv, sqlite3
import pandas as pd

from constants import SQLITE_DIRECTORY, CSV_DIRECTORY
db_file_name = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')

conn = sqlite3.connect(db_file_name)
curr = conn.execute("SELECT * from patient_data where patient_sex = 'M' AND patient_covid_status=1")
res = curr.fetchall()

patient_site_uid = str(res[0][0])

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
print(patient_site_uid)
curr = conn.execute("SELECT * from imaging_data where imaging_modality='XR' AND imaging_site='chest'")
curr = conn.execute("SELECT * from imaging_data where patient_site_uid='" + patient_site_uid + "'")

res = curr.fetchall()
print(len(res))