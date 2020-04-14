# CoviDB: a multimedia database of patients with COVID-19

## 1.Installing

### 1.1 Prerequisites

* Anaconda / Python 3 environment
* NumPy: `conda install numpy`
* Pillow: `conda install pillow`
* LibJPEG and OpenJPEG: `conda install -c conda-forge openjpeg jpeg`
* GDCM: `conda install gdcm -c conda-forge`
* PyDicom: `conda install -c conda-forge pydicom`

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

The following CSV files will be generated:

*`patient_data.csv`: contains temporally invariant patient data
*`imaging_data.csv`: contains a list of imaging studies associated with patients
*`slice_data.csv`: contains a list of DICOM files associated with imaging studies
*`lab_data.csv`: contains a list of lab tests
*`pcr_data.csv`: contains a list of PCR tests
*`micro_data.csv`: contains a list of microbiological data

After the CSV files are generated, these will be imported into an SQLite database. This database will contain one table for each of the CSV files. 

## 3. Using the database

### 3.1 Import the SQLite file

