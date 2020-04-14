import hashlib
from constants import DICOM_ID_MAP, PATIENT_GLOBAL_SALT, PATIENT_SITE_SALT

def get_patient_site_uid_from_dicom_id(dicom_id):
  dicom_id_str = str(dicom_id)
  if not dicom_id_str in DICOM_ID_MAP:
    print('Fatal: DICOM ID not found in map.')
    exit()
  return get_patient_uid(DICOM_ID_MAP[dicom_id_str])

def generate_patient_site_uid(patient_mrn):
  
  patient_mrn = str(patient_mrn)

  return hashlib.pbkdf2_hmac('sha256', 
    patient_mrn.encode('utf-8'), 
    PATIENT_SITE_SALT.encode('utf-8'), 10000).hex()

def generate_patient_uid(patient_ramq):

  patient_ramq = str(patient_ramq)

  return hashlib.pbkdf2_hmac('sha256', 
    patient_ramq.encode('utf-8'), 
    PATIENT_GLOBAL_SALT.encode('utf-8'), 10000).hex()

def generate_slice_study_uid(study_uid):

  return hashlib.pbkdf2_hmac('sha256', 
    study_uid.encode('utf-8'), 
    PATIENT_SITE_SALT.encode('utf-8'), 10000).hex()[0:16]


def generate_slice_series_uid(series_uid):

  return hashlib.pbkdf2_hmac('sha256', 
    series_uid.encode('utf-8'), 
    PATIENT_SITE_SALT.encode('utf-8'), 10000).hex()[0:16]

