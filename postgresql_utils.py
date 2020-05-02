from connect_postgres import connectDB_returnDF

def sql_query(query):
  return connectDB_returnDF(str(query))

def list_columns(table_name):
  print(connectDB_returnDF("SELECT COLUMN_NAME, DATA_TYPE from " + 
    "INFORMATION_SCHEMA.COLUMNS IC where TABLE_NAME='"+str(table_name)+"'"))

def list_tables(table_schema, table_name):
  df = connectDB_returnDF("SELECT * from information_schema.tables " + \
    "WHERE table_schema = 'public' AND table_name LIKE '%" + table_name + "%'")
  for row in df.iterrows():
    print(row)

#list_tables('public', 'urgchum_episod_note_inf')


#df2 = sql_query(
# "SELECT * from public.urgchum_episod_note_inf WHERE " + 
# "note LIKE '%covid%' LIMIT 100 "
#)

#print(df2.iloc[0])

#exit()
#exit()
#list_columns('lb_mic_sus_med_result')

#df2 = sql_query(
#  "SELECT * from dw_v01.oacis_rd INNER JOIN " +
#  "dw_v01.oacis_rdreport_text ON " +
#  "oacis_rd.sid = oacis_rdreport_text.sid " +
#  "WHERE oacis_rd.dictationdtm > '2018-01-01' AND " +
#  "oacis_rd.codes_examens LIKE '%POUMON%' " +
#  "AND report_ck LIKE '%endotrach%' LIMIT 100"
#)

# '8100 POUMONS' 
# '8100 POUMONS (2 INCIDENCE' 
# '8106 POUMONS (3 INCIDENCE'

#print(df2.iloc[0].report_ck)

#
#list_tables('dw_v01', 'rdreport')
