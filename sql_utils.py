from connect_postgres import connectDB_returnDF

def sql_query(query):
  return connectDB_returnDF(str(query))

def list_columns(table_name):
  print(connectDB_returnDF("SELECT COLUMN_NAME, DATA_TYPE from " + 
    "INFORMATION_SCHEMA.COLUMNS IC where TABLE_NAME='"+str(table_name)+"'"))

def list_tables():
  # df = connectDB_returnDF('SELECT * from dw_test.ci_sejhosp_lit_live LIMIT 100')
  
  print('\n***** DICOM-related tables ')
  df = connectDB_returnDF("SELECT * from dw_test.dw_dicom_citadel_idx WHERE dtm_sign > '2015-01-01'")
  print(df)


#list_tables()