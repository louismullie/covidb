from connect_postgres import connectDB_returnDF

import pandas as pd
pd.set_option('display.max_rows', 250)

def sql_query(query):
  return connectDB_returnDF(str(query))

def list_columns(table_name):
  print(connectDB_returnDF("SELECT COLUMN_NAME, DATA_TYPE from " + 
    "INFORMATION_SCHEMA.COLUMNS IC where TABLE_NAME='"+str(table_name)+"'"))

def list_tables(table_schema, table_name):
  df = connectDB_returnDF("SELECT * from information_schema.tables " + \
    "WHERE table_schema = 'public' AND table_name LIKE '%" + table_name + "%'")
  
  for i, row in df.iterrows():
    print(row)

#list_tables('public', 'zrrob')


#df = sql_query(
# "SELECT * FROM dw_v01.citadel_chum_unitesoins "
#)

#print(df)

#for i, row in df.iterrows():
#  print([x for x in row if x is not None])
#exit()
#exit()
#list_columns('oacis_zrrob')

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
