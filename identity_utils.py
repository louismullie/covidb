import hashlib

patient_uid_salt = '|.@Zbyi*7@?OBKZ4'
patient_site_uid_salt = ':/o"3yR<>|9ue~{/'

def generate_patient_site_uid(patient_mrn):
  
  patient_mrn = str(patient_mrn)

  return hashlib.pbkdf2_hmac('sha256', 
    patient_mrn.encode('utf-8'), 
    patient_site_uid_salt.encode('utf-8'), 10000).hex()

def generate_patient_uid(patient_ramq):

  patient_ramq = str(patient_ramq)

  return hashlib.pbkdf2_hmac('sha256', 
    patient_ramq.encode('utf-8'), 
    patient_uid_salt.encode('utf-8'), 10000).hex()

def generate_slice_study_uid(study_uid):

  return hashlib.pbkdf2_hmac('sha256', 
    study_uid.encode('utf-8'), 
    patient_site_uid_salt.encode('utf-8'), 10000).hex()[0:16]


def generate_slice_series_uid(series_uid):

  return hashlib.pbkdf2_hmac('sha256', 
    series_uid.encode('utf-8'), 
    patient_site_uid_salt.encode('utf-8'), 10000).hex()[0:16]