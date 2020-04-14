#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, BLOB_DIRECTORY, \
  LIVE_SHEET_FILENAME, DICOM_ID_MAP_FILENAME, CSV_DIRECTORY, DICOM_DIRECTORY
from sql_utils import sql_query
from file_utils import read_csv, write_csv
from image_utils import equalize_histogram
from identity_utils import generate_slice_study_uid, generate_slice_series_uid, get_patient_site_uid_from_dicom_id
from time_utils import get_hours_between_datetimes
from dicom_utils import read_dcm, window_level
from pydicom import Dataset

series_counter = {}
slice_data_rows = []

for r, d, files in os.walk(DICOM_DIRECTORY):
  for f in files:
    dicom_file_path = os.path.join(r, f)
    dicom = read_dcm(dicom_file_path)

    transfer_syntax = dicom.file_meta.TransferSyntaxUID
    
    pixel_data = None

    try:
      pixel_data = dicom.get('PixelData')
      if pixel_data == None:
        raise
    except:
      if 'KO' not in dicom_file_path \
        and 'PR' not in dicom_file_path \
        and 'SR' not in dicom_file_path:
          if DEBUG: print('Skipping slice %s: no pixel data' % dicom_file_path)
      else: continue # skip metadata files
    
    pixel_array = None

    try:
      pixel_array = dicom.pixel_array
    except:
      if DEBUG:
        print('Skipping slices: cannot read pixel data')
        print(transfer_syntax)
      continue

    patient_mrn = str(dicom.get('PatientID'))
    slice_study_id = dicom.get('StudyInstanceUID')
    slice_series_id = dicom.get('SeriesInstanceUID')
    slice_view_position = dicom.get('ViewPosition')
    slice_patient_position = dicom.get('PatientPosition')
    slice_image_orientation = dicom.get('ImageOrientationPatient')
    slice_image_position = dicom.get('ImagePositionPatient')
    slice_window_center = dicom.get('indowCenter')
    slice_window_width = dicom.get('WindowWidth')
    slice_pixel_spacing = dicom.get('PixelSpacing')
    slice_thickness = dicom.get('SliceThickness')
    slice_rows = dicom.get('Rows')
    slice_columns = dicom.get('Columns')
    slice_rescale_intercept = dicom.get('RescaleIntercept')
    slice_rescale_slope = dicom.get('RescaleSlope')

    if slice_rescale_intercept is None: slice_rescale_intercept = 0
    if slice_rescale_slope is None: slice_rescale_slope = 1

    pixel_array = slice_rescale_slope * \
      pixel_array.astype(np.float64) + slice_rescale_intercept
    
    #pixel_array -= np.min(pixel_array)
    #pixel_array /= np.max(pixel_array)
    #print(pixel_array)
    #pixel_array = equalize_histogram(pixel_array)
    pixel_array = (pixel_array).astype(np.int16)

    #from PIL import Image
    #img = Image.fromarray(pixel_array)
    #img.save("test.jpeg")

    #exit()
    #print(rescaled_pixel_array)
    study_uid = generate_slice_study_uid(slice_study_id)
    series_uid = generate_slice_series_uid(slice_series_id)

    if study_uid not in series_counter:
      series_counter[study_uid] = {}

    if series_uid not in series_counter[study_uid]:
      series_counter[study_uid][series_uid] = 0
    else:
      series_counter[study_uid][series_uid] += 1

    slice_num = series_counter[study_uid][series_uid]

    slice_data_file_path = os.path.join(BLOB_DIRECTORY, study_uid, series_uid)
    slice_data_file_name = 'slice_' + str(slice_num) + '.csv'

    os.makedirs(slice_data_file_path, exist_ok=True)

    slice_data_uri = os.path.join(
      slice_data_file_path, slice_data_file_name)

    slice_data_rows.append(['' if x == None else x for x in [
      slice_study_id,
      slice_series_id,
      slice_data_uri,
      slice_view_position,
      slice_patient_position,
      slice_image_orientation,
      slice_image_position,
      slice_window_center,
      slice_window_width,
      slice_pixel_spacing,
      slice_thickness,
      slice_rows,
      slice_columns,
      slice_rescale_intercept,
      slice_rescale_slope
    ]])
    print(pixel_array)
    df = pd.DataFrame(data=pixel_array)
    csv = df.to_csv()
    with open(slice_data_uri, 'w') as data_file:
      data_file.write(csv)
    exit()

print('Total rows: %d' % len(slice_data_rows))
write_csv(TABLE_COLUMNS['slice_data'], slice_data_rows, 
  os.path.join(CSV_DIRECTORY, 'slice_data.csv'))