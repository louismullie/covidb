# CoviDB: a multimedia database of patients with COVID-19

## 1.Installing

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

### 1.3 Suggested folder structure

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

```python
from constants import SQLITE_DIRECTORY
db_file_name = os.path.join(SQLITE_DIRECTORY, 'covid_v1.0.0.db')

conn = sqlite3.connect(db_file_name)

# Fetch an imaging study from imaging data
curr = conn.execute("SELECT * from imaging_data where imaging_site = 'chest' LIMIT 1")
study = curr.fetchone()

# Retrieve the patient site UID of interest
imaging_data_id = study.imaging_data_id

# Retrieve the DICOM slices for the imaging study
curr = conn.execute("SELECT * from slice_data where imaging_data_id = '%s'" % imaging_data_id)
slices = curr.fetchall()

# Retrieve the data files for the imaging study
for slice in slices:
  data_frame = pd.DataFrame.from_csv(slice.slice_data_uri)
```

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
