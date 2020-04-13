import os, csv, sqlite3
import pandas as pd

from constants import SQLITE_DIRECTORY, CSV_DIRECTORY
db_file_name = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')

conn = sqlite3.connect(db_file_name)
curr = conn.execute("SELECT * from patient_data where patient_sex = 'M' AND patient_covid_status=1")
res = curr.fetchall()
print(len(res))
print(res[0])

patient_site_uid = res[0][0]

curr = conn.execute("SELECT * from lab_data where lab_name = 'Lympho #' AND patient_site_uid='" + patient_site_uid + "'")
res = curr.fetchall()
print(len(res))
print(res[0])

# Fetch all positive PCR tests
curr = conn.execute("SELECT * from pcr_data where pcr_name LIKE '%COVID%' AND patient_site_uid='" + patient_site_uid + "'")
res = curr.fetchall()
print(res)