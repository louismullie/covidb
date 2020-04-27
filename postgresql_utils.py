from connect_postgres import connectDB_returnDF

def sql_query(query):
  return connectDB_returnDF(str(query))

def list_columns(table_name):
  print(connectDB_returnDF("SELECT COLUMN_NAME, DATA_TYPE from " + 
    "INFORMATION_SCHEMA.COLUMNS IC where TABLE_NAME='"+str(table_name)+"'"))

def list_tables(table_schema, table_name):
  df = connectDB_returnDF("SELECT * from information_schema.tables " + \
    "WHERE table_name LIKE '%" + table_name + "%'")
  for row in df.iterrows():
    print(row)

#list_tables('dw_v01', 'lb_mic_sus_med_result')
#list_columns('lb_mic_sus_med_result')

##df = sql_query("SELECT * from dw_test.orcl_cichum_sejurg_live LIMIT 1")
#episode_id = df.iloc[0].noadm
#print(episode_id)

#df2 = sql_query('SELECT * from public.lb_mic_sus_med_result LIMIT 1')
#print(df2.iloc[0])


