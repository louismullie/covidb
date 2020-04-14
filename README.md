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


