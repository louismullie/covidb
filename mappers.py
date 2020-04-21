import re

def map_patient_ramq(ramq):
  ramq_str = str(ramq).strip()
  if not re.match("[A-Z]{4}[0-9]{8}", ramq_str):
    if ramq_str != '':
      return False
  return ramq_str

def map_patient_covid_status(status):
  status_str = str(status).strip().lower()
  if status_str[0] == 'p':   return 1 # Positif
  elif status_str[0] == 'n': return 2 # Negatif
  elif status_str[0] == 'e': return 3 # En attente
  elif status_str[0] == 't': return 4 # Test annule
  elif status_str[0] == 'a': return 4 # Annule
  elif status_str[0] == 'r': return 4 # Rapport numerise
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

#def map_lab_result(lab_result):
  
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

def map_lab_result_value(result_string):
  result_string = str(result_string) \
    .replace('<', '') \
    .replace('>', '') \
    .replace(',', '.') \
    .strip()
  if result_string == '':
    return ''
  else:
    return float(result_string)

def map_culture_sample_site(desc):
  if desc is None: return 8
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

def map_culture_growth_value(value):
  if value is None: return ''
  value_str = str(value).strip().lower()
  if value_str == 'pos': return 1
  elif value_str == 'neg': return 0
  else: 
    print('Unrecognized growth result')
    return ''

def map_episode_unit_type(unit_code):
  unit_code_str = str(unit_code)
  if '10S' in unit_code_str or '10N' in unit_code_str:
    return 5
  else: 
    return 3

def map_observation_name(name):
  if name == 'FIO2': return 'fraction_inspired_oxygen'
  if name == 'Sat O2 Art': return 'arterial_oxygen_saturation'
  if name == 'Température': return 'temperature'
  return 'unknown'