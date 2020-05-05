import os, csv, sqlite3
import pandas as pd

from constants import SQLITE_DIRECTORY, CSV_DIRECTORY, TABLE_COLUMNS
from identity_utils import generate_patient_uid, generate_patient_site_uid

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')

with open(db_file_name, 'w') as db_file:
  db_file.write('')

conn = sqlite3.connect(db_file_name)

schema_sql = None
with open('create_schema_sqlite.sql') as schema_sql_file:
  schema_sql = schema_sql_file.read()

if schema_sql is None:
  print('Error: schema SQL is empty')

schema_blocks = schema_sql.split(';')

for schema_block in schema_blocks:
  conn.execute(schema_block + ';')

curr = conn.cursor()

for table_name in TABLE_COLUMNS:
  file_uri = os.path.join(CSV_DIRECTORY, table_name + '.csv')
  df = pd.read_csv(file_uri)
  #if 'patient_site_uid' in df.columns:
  #  df['patient_site_uid'] = df['patient_site_uid'].map(lambda x: 
  #  generate_patient_site_uid(x))
  #if 'patient_uid' in df.columns:
  #  df['patient_uid'] = df['patient_uid'].map(lambda x: 
  #    generate_patient_uid(x))
  df.rename({ 'patient_mrn': 'patient_site_uid', 'patient_ramq': 'patient_uid'})
  sql = df.to_sql(table_name, conn, if_exists='replace', index=False) 

  conn.commit()
conn.close()