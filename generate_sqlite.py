import sqlite3 as sqlite
import os
from constants import SQLITE_DIRECTORY

db_file = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')
conn = sqlite.connect(db_file)

schema_sql = None
with open('create_schema.sql') as schema_sql_file:
  schema_sql = schema_sql_file.read()

if schema_sql is None:
  print('Error: schema SQL is empty')

conn.execute(schema_sql)