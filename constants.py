import pandas as pd
pd.set_option('display.max_rows', 250)

DEBUG = True

LIVE_SHEET_FILENAME = "/var/www/html/mchasse/covid19/data_all.csv"
CSV_DIRECTORY = "/data8/projets/Mila_covid19/output/csv"
BLOB_DIRECTORY = "/data8/projets/Mila_covid19/output/blob"
DICOM_DIRECTORY = "/data8/projets/Mila_covid19/data/covid_citadel_pacs"
SQLITE_DIRECTORY = "/data8/projets/Mila_covid19/output/sqlite"
CODE_DIRECTORY = "/data8/projets/Mila_covid19/code/lmullie/git_Mila_covid19"

TABLE_COLUMNS = {

  'patient_data': [
    'patient_site_uid', 'patient_uid', 'pcr_sample_time', 
    'patient_site_code', 'patient_transfer_site_code', 
    'patient_covid_status', 'patient_age', 'patient_birth_sex'
  ],

  'lab_data': [
    'patient_site_uid', 'lab_name', 'lab_sample_site', 'lab_sample_time', 
    'lab_result_time', 'lab_result_value', 'lab_result_units'
  ],

  'pcr_data': [
    'patient_site_uid', 'pcr_name', 'pcr_sample_site', 'pcr_sample_time', 
    'pcr_result_time', 'pcr_result_value', 'patient_covid_status'
  ],

  'micro_data': [
    'patient_site_uid', 'micro_name', 'micro_sample_site', 'micro_sample_time', 
    'micro_result_time', 'micro_result_value'
  ],
  'imaging_data': [
    'patient_site_uid', 'imaging_accession_number', 'imaging_modality'
  ],
  'slice_data': [
    'patient_site_uid', 'slice_study_id', 'slice_series_id', 'slice_data_uri', 
    'slice_view_position', 'slice_patient_position', 'slice_image_orientation',
    'slice_image_position', 'slice_window_center', 'slice_window_width', 
    'slice_pixel_spacing', 'slice_thickness', 'slice_rows', 'slice_columns',
    'slice_rescale_intercept', 'slice_rescale_slope'
  ]
}
