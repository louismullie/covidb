# CoviDB: a multimodal repository of patients with COVID-19

## 1. Installing

### 1.1 Prerequisites

Ensure the following is installed before using this repository.

- Anaconda / Python 3 environment
- NumPy: `conda install numpy`
- Pillow: `conda install pillow`
- LibJPEG and OpenJPEG: `conda install -c conda-forge openjpeg jpeg`
- GDCM: `conda install gdcm -c conda-forge`
- PyDicom: `conda install -c conda-forge pydicom`

### 1.2 Clone Git repo

`git clone git@github.com:louismullie/covidb.git`

### 1.3 Folder structure

The following is a suggested project folder. Data and output files are located in subfolders of the parent folder. 

```
/covid19
  /code --> this repository
  /data
    /dicom_ids
    /dicom_files
  /output
    /csv
    /blob
    /sqlite
```

The structure of the `blob` folder is as follows:

```
/[imaging_accession_uid]
  /[study_instance_uid]
    /[series_instance_uid]
      .csv files containing pixel data
```

### 1.4 Edit settings

Edit the following settings in constants.py:

## 2. Generating database

### 2.1 Run generation script

`python_generate_covidb.py`

### 2.2 Output description

The following CSV files will be generated inside `CSV_DIRECTORY`, as defined in `constants.py`:

- Patient data - `patient_data.csv`: contains temporally invariant patient data
- Imaging data - `imaging_data.csv`: contains a list of imaging studies associated with patients
- Slice data - `slice_data.csv`: contains a list of DICOM files associated with imaging studies
- Laboratory data - `lab_data.csv`: contains a list of lab tests
- PCR data - `pcr_data.csv`: contains a list of PCR tests
- Micro data - `micro_data.csv`: contains a list of microbiological data

After the CSV files are generated, these will be imported into an SQLite database. This database will contain one table for each of the CSV files. 

The SQLite database will be generated as `covid_v1.0.0.db` inside `SQLITE_DIRECTORY`, as defined in `constants.py`.

## 3. Using the database

### 3.1 Etablishing a connection

```python
from constants import SQLITE_DIRECTORY
db_file_name = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')

conn = sqlite3.connect(db_file_name)
curr = conn.execute("SELECT * from patient_data where patient_sex = 'M' AND patient_covid_status=1")
res = curr.fetchall()
```

### 3.2 Fetching the data for a DICOM file

First, fetch an imaging study and retrieve its ID (`imaging_data_id`).

```python
from constants import SQLITE_DIRECTORY
db_file_name = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')

conn = sqlite3.connect(db_file_name)

# Fetch an imaging study from imaging data
curr = conn.execute("SELECT * from imaging_data where imaging_site = 'chest' LIMIT 1")
study = curr.fetchone()

# Retrieve the patient site UID of interest
imaging_accession_uid = study.imaging_accession_uid
```

Next, retrieve the slice(s) associated with the imaging study, and retrieve the location on disk of their pixel data files  (`slice_data_uri`).

```python
# Retrieve the DICOM slices for the imaging study
curr = conn.execute("SELECT * from slice_data where imaging_accession_uid = '%s'" % imaging_accession_uid)
slices = curr.fetchall()

# Retrieve the data files for the imaging study
for slice in slices:
  data_frame = pd.DataFrame.from_csv(slice.slice_data_uri)
```

The file on disk representing the pixel data is a matrix of 32-bit integers, stored as a CSV file in ASCII text encoding, with commas as separators and no quotes around fields. This can be read directly into a Pandas data frame (as shown above), or converted to a NumPy array (see below).

### 3.3 Displaying a DICOM file
```python
import numpy as np
import matplotlib.pyplot as plt
from image_utils import equalize_histogram

imge_data_frame = pd.DataFrame.from_csv(dicom_data_uri)
image_numpy_array = data_frame.to_numpy()

# Optional - use equalization method best suited for use
image_contrast_adjusted = equalize_histogram(numpy_array)

plt.imshow(image_contrast_adjusted, cmap="gray")
plt.show()
```

## 4. Available tables and columns

The list of available tables and columns, as of version 1.0.0, includes:

```python
TABLE_COLUMNS = {

  'patient_data': [
    'patient_site_uid', 'patient_uid', 'pcr_sample_time', 
    'patient_site_code', 'patient_covid_status', 'patient_age', 
    'patient_sex', 'patient_vital_status', 'patient_was_admitted'
  ],

  'episode_data': [
    'patient_site_uid', 'episode_admission_uid', 
    'episode_unit_type', 'episode_start_time', 
    'episode_end_time', 'episode_description', 
    'episode_duration_hours'
  ],

  'diagnosis_data': [
    'patient_site_uid', 'episode_id', 'diagnosis_type',
    'diagnosis_name', 'diagnosis_icd_code', 'diagnosis_time'
  ],

  'drug_data': [
    'patient_site_uid', 'drug_name', 
    'drug_start_time', 'drug_end_time',
    'drug_frequency', 'drug_roa'
  ],

  'lab_data': [
    'patient_site_uid', 'lab_name', 'lab_sample_site', 
    'lab_sample_time', 'lab_result_time', 
    'lab_result_status', 'lab_result_units',
    'lab_result_string', 'lab_result_value'
  ],

  'observation_data': [
    'patient_site_uid', 'observation_name', 
    'observation_time', 'observation_value', #'observation_units'
  ],

  'pcr_data': [
    'patient_site_uid', 'pcr_name', 'pcr_sample_site', 'pcr_sample_time', 
    'pcr_result_time', 'pcr_result_value', 'pcr_result_status'
  ],

  'culture_data': [
    'patient_site_uid', 'culture_type', 'culture_specimen_type', 'culture_sample_time', 
    'culture_result_time', 'culture_growth_positive', 'culture_result_status'
  ],

  'imaging_data': [
    'patient_site_uid', 'imaging_accession_uid', 'imaging_modality', 'imaging_site',
    'imaging_acquisition_time'
  ],

  'slice_data': [
    'patient_site_uid', 'imaging_accession_uid', 'slice_study_instance_uid', 
    'slice_series_instance_uid', 'slice_data_uri', 'slice_view_position', 
    'slice_patient_position', 'slice_image_orientation', 'slice_image_position', 
    'slice_window_center', 'slice_window_width', 'slice_pixel_spacing', 
    'slice_thickness', 'slice_rows', 'slice_columns', 'slice_rescale_intercept', 
    'slice_rescale_slope'
  ]
  
}
```
