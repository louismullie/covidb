from file_utils import read_csv

import pandas as pd

pd.set_option("display.max_rows", 250)

DEBUG = True

PATIENT_GLOBAL_SALT = "1fd5789d7ef4287fd8acfc765061e10eb3e7c093ff9150978695fb83692e4a87d55c4abf83c7ad9bcc3305ab03a4d28a5c404db6b84886c1665f949215e75a2b"
PATIENT_SITE_SALT = "243460170aec12b2cb4ce6e92d1293ebe8bbc83b4a860681ecfd4b653961f253fc3cb7ae833de5a4faca2d98ed9789e061e95aea7335901e6c84c7c05feee85f"
LIVE_SHEET_FILENAME = "/var/www/html/mchasse/covid19/data_all.csv"
CSV_DIRECTORY = "/data/chum/041620/csv_citadel_hashed"
BLOB_DIRECTORY = "/data8/projets/Mila_covid19/output/staging/blob"
DICOM_MAP_FILENAME = "/data8/projets/Mila_covid19/data/patient_infos/dicom_id_map.csv"
DICOM_DIRECTORY = "/data/chum/041620/covid_citadel_pacs"
SQLITE_DIRECTORY = "/data/chum/041620/sqlite"
CODE_DIRECTORY = "/home/sinhakou-local/mlp/covidb"

# dicom_id_map_rows = read_csv(DICOM_MAP_FILENAME)
# DICOM_PATIENT_ID_MAP = {}
# DICOM_STUDY_ID_MAP = {}

# for dicom_id_map_row in dicom_id_map_rows:
#   if len(dicom_id_map_row) == 0: continue
#   DICOM_PATIENT_ID_MAP[str(dicom_id_map_row[2])] = str(dicom_id_map_row[0])
#   DICOM_STUDY_ID_MAP[str(dicom_id_map_row[2])] = str(dicom_id_map_row[1])

TABLE_COLUMNS = {
    "patient_data": [
        "patient_site_uid",
        "patient_uid",
        "pcr_sample_time",
        "patient_site_code",
        "patient_transfer_site_code",
        "patient_covid_status",
        "patient_age",
        "patient_sex",
    ],
    "episode_data": [
        "patient_site_uid",
        "episode_unit_type",
        "episode_start_time",
        "episode_end_time",
        "episode_description",
    ],
    "diagnosis_data": ["patient_site_uid", "diagnosis_type", "diagnosis_time"],
    "lab_data": [
        "patient_site_uid",
        "lab_name",
        "lab_sample_site",
        "lab_sample_time",
        "lab_result_time",
        "lab_result_value",
        "lab_result_units",
    ],
    "pcr_data": [
        "patient_site_uid",
        "pcr_name",
        "pcr_sample_site",
        "pcr_sample_time",
        "pcr_result_time",
        "pcr_result_value",
    ],
    "micro_data": [
        "patient_site_uid",
        "micro_name",
        "micro_sample_site",
        "micro_sample_time",
        "micro_result_time",
        "micro_result_value",
    ],
    "imaging_data": [
        "patient_site_uid",
        "imaging_accession_uid",
        "imaging_modality",
        "imaging_site",
    ],
    "slice_data": [
        "patient_site_uid",
        "imaging_accession_uid",
        "slice_study_instance_uid",
        "slice_series_instance_uid",
        "slice_data_uri",
        "slice_view_position",
        "slice_patient_position",
        "slice_image_orientation",
        "slice_image_position",
        "slice_window_center",
        "slice_window_width",
        "slice_pixel_spacing",
        "slice_thickness",
        "slice_rows",
        "slice_columns",
        "slice_rescale_intercept",
        "slice_rescale_slope",
    ],
}
