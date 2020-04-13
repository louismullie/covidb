#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

"""
import os
import numpy as np
import pandas as pd

from constants import DEBUG, TABLE_COLUMNS, LIVE_SHEET_FILENAME, SLICE_DATA_DIRECTORY
from sql_utils import sql_query, list_columns
from file_utils import write_csv
from time_utils import get_hours_between_datetimes
from mappers import map_patient_covid_status, map_patient_age, map_patient_birth_sex
from dicom_utils import read_dcm

# Currently handled in patient_data - to be moved here