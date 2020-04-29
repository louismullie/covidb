import hashlib
from constants import DICOM_PATIENT_ID_MAP, DICOM_STUDY_ID_MAP, \
                      PATIENT_GLOBAL_SALT, PATIENT_SITE_SALT

def get_patient_mrn_from_dicom_study_id(dicom_study_id):
  dicom_id_str = str(dicom_study_id)
  if not dicom_id_str in DICOM_PATIENT_ID_MAP:
    print('Fatal: DICOM ID not found in map.')
    exit()
  patient_mrn = DICOM_PATIENT_ID_MAP[dicom_id_str]
  if patient_mrn[0] == 'S': 
    patient_mrn = patient_mrn[1:]
  return patient_mrn

def get_accession_number_from_dicom_study_id(dicom_study_id):
  dicom_id_str = str(dicom_study_id)
  if not dicom_id_str in DICOM_STUDY_ID_MAP:
    print('Fatal: DICOM ID not found in map.')
    print(dicom_id_str)
    exit()
  return DICOM_STUDY_ID_MAP[dicom_id_str]

def generate_patient_site_uid(patient_mrn):
  
  patient_mrn = str(patient_mrn)

  return hashlib.pbkdf2_hmac('sha512', 
    patient_mrn.encode('utf-8'), 
    PATIENT_SITE_SALT.encode('utf-8'), 100000).hex()

def generate_patient_uid(patient_ramq):

  patient_ramq = str(patient_ramq)

  return hashlib.pbkdf2_hmac('sha512', 
    patient_ramq.encode('utf-8'), 
    PATIENT_GLOBAL_SALT.encode('utf-8'), 100000).hex()

def generate_accession_uid(accession_number):

  return hashlib.pbkdf2_hmac('sha512', 
    accession_number.encode('utf-8'), 
    PATIENT_SITE_SALT.encode('utf-8'), 100000).hex()

def generate_slice_study_uid(study_uid):

  return hashlib.pbkdf2_hmac('sha512', 
    study_uid.encode('utf-8'), 
    PATIENT_SITE_SALT.encode('utf-8'), 100000).hex()

def generate_slice_series_uid(series_uid):

  return hashlib.pbkdf2_hmac('sha512', 
    series_uid.encode('utf-8'), 
    PATIENT_SITE_SALT.encode('utf-8'), 100000).hex()

