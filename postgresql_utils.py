from connect_postgres import connectDB_returnDF

def sql_query(query):
  return connectDB_returnDF(str(query))

def list_columns(table_name):
  print(connectDB_returnDF("SELECT COLUMN_NAME, DATA_TYPE from " + 
    "INFORMATION_SCHEMA.COLUMNS IC where TABLE_NAME='"+str(table_name)+"'"))

def list_tables(table_schema, table_name):
  df = connectDB_returnDF("SELECT * from information_schema.tables " + \
    "WHERE table_schema='" + table_schema + \
    "' AND table_name LIKE '%" + table_name + "%'")
  for row in df.iterrows():
    print(row)

#list_tables('dw_v01', 'urg')