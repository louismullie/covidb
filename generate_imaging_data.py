#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, COLUMNS, LIVE_SHEET_FILENAME, SLICE_DATA_DIRECTORY
from sql_utils import sql_query, list_columns
from file_utils import write_csv
from time_utils import get_hours_between_datetimes
from mappers import map_patient_covid_status, map_patient_age, map_patient_birth_sex
from dicom_utils import read_dcm

slice_data_rows = []

for r, d, files in os.walk(SLICE_DATA_DIRECTORY):
  for f in files:
    dicom = read_dcm(os.path.join(r, f))

    slice_study_id = dicom.get('StudyInstanceUID')
    slice_series_id = dicom.get('SeriesInstanceUID')
    slice_data_uri = 'path'
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
    slice_rescale_slope = dicom.get('Slope')

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


print('Total rows: %d' % len(slice_data_rows))
write_csv(COLUMNS['slice_data'], slice_data_rows, './csv/slice_data.csv')