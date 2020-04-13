import os, csv, sqlite3
import pandas as pd

from constants import SQLITE_DIRECTORY, CSV_DIRECTORY

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')

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

file_uri = os.path.join(CSV_DIRECTORY, 'patient_data.csv')
df = pd.read_csv(file_uri)
sql = df.to_sql('patient_data', conn, if_exists='replace', index=False)
print(sql)
#conn.commit()
conn.close()