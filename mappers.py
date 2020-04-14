import re

def map_patient_ramq(ramq):
  ramq_str = str(ramq).strip()
  if not re.match("[A-Z]{4}[0-9]{8}", ramq_str):
    if ramq_str != '':
      return False
  return ramq_str

def map_patient_covid_status(status):
  if status[0] == 'P':   return 1 # Positif
  elif status[0] == 'N': return 2 # Negatif
  elif status[0] == 'E': return 5 # En attente
  elif status[0] == 'T': return 6 # Test annule
  elif status[0] == 'A': return 6 # Annule
  elif status[0] == 'R': return 5 # Rapport numerise
  else: raise Exception('Invalid COVID status: %s' % status)

def map_patient_age(age):
  age_str = str(age).strip()
  if age_str == '': return ''
  age_parsed = int(age_str)
  if age_parsed < 0 or age_parsed > 122:
    raise Exception('Invalid age: %s' % age)
  return age_parsed

def map_patient_sex(sex):
  sex_parsed = (str(sex).strip()).lower()
  if sex_parsed == '': return ''
  if sex_parsed == 'm': return 'M'
  elif sex_parsed == 'f': return 'F'
  elif sex_parsed == 'x': return 'X'
  else: raise Exception('Invalid birth sex: %s' % sex)

def map_lab_sample_site(code):
  code = (str(code).strip()).lower()
  if 'veineux' in code: return 1
  elif 'art' in code: return 2
  elif 'urine' in code: return 4
  elif 'autres' in code: return 5
  elif 'none' in code: return 6
  else:
    print('Unrecognized sample site: ' + code)
    return 5

def map_pcr_sample_site(code):
  code = (str(code).strip()).lower()
  if 'couvillon' in code: return 1
  elif 'urine' in code: return 8
  elif 'sang' in code: return 9
  elif 'micro' in code: return 10
  elif 'autres' in code: return 10
  elif 'none' in code: return 10
  else:
    print('Unrecognized sample site: ' + code)
    return 5

def map_micro_sample_site(desc):
  desc = (str(desc).strip()).lower()
  if 'moculture' in desc: return 1
  elif 'sang' in desc: return 1
  elif 'urine' in desc: return 2
  elif 'lavage broncho-a' in desc: return 3
  elif 'intravasculaire' in desc: return 4
  elif 'bronchique' in desc: return 5
  elif 'expectoration' in desc: return 6
  elif 'pus profond' in desc: return 7
  elif 'cimen micro' in desc: return 8
  elif 'autres' in desc: return 8
  else:
    print('Unrecognized sample site: ' + desc)
    return 8