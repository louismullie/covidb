import re
import numpy as np
from constants import DRUG_NAME_MAP, DRUG_ROUTE_MAP, LAB_NAMES_MAP
from lang_utils import transliterate_string
from time_utils import get_hours_between_datetimes

def map_string_lower(str_value):
  if str_value is None: return ''
  return str(str_value).lower().strip()

def map_time(time):
  if time is None: return ''
  if str(time).lower() == 'nat': return ''
  return str(time).strip().lower()

def map_float_value(float_value):
  if float_value is None: return ''
  float_value_str = str(float_value).strip()
  float_value_str = float_value_str.replace(',', '.')
  if float_value_str == 'nan': return ''
  try:
    val = float(float_value_str)
    return str(val)
  except:
    print('Invalid float value: %s' % float_value_str)
    exit()

def map_patient_ramq(ramq):
  ramq_str = str(ramq).strip()
  if not re.match("[A-Z]{4}[0-9]{8}", ramq_str):
    return ''
  return ramq_str

def map_patient_covid_status(status):
  status_str = str(status).strip().lower()
  if status_str == 'positif':   return 'positive'
  elif status_str == 'négatif': return 'negative'
  elif status_str == 'en attente': return 'pending'
  elif status_str == 'test annulé': return 'unknown'
  elif status_str == 'annulé': return 'unknown'
  elif status_str == 'rapp. numérisé': return 'unknown'
  elif status_str == 'non valide': return 'unknown'
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
  if sex_parsed == 'm': return 'male'
  elif sex_parsed == 'f': return 'female'
  elif sex_parsed == 'x': return 'unspecified'
  else: raise Exception('Invalid birth sex: %s' % sex)

def map_lab_name(name):
  name_str = str(name).strip().lower()
  if name_str in LAB_NAMES_MAP:
    return LAB_NAMES_MAP[name_str]
  else:
    print('Invalid lab name: %s' % name_str)
  
def map_lab_sample_site(name, code):
  name_str = str(name).strip().lower()
  code_str = (str(code).strip()).lower()

  if 'art ecmo' in name_str: return 'ecmo_arterial_blood'
  if 'vein ecmo' in name_str: return 'ecmo_venous_blood'
  if 'adbd' in name_str: return 'capillary_blood'
  if 'vein' in code_str: return 'venous_blood'
  if 'vein' in name_str: return 'venous_blood'
  if '(v)' in name_str: return 'venous_blood'
  if 'art' in code_str: return 'arterial_blood'
  if 'art' in name_str: return 'arterial_blood'
  if 'urine' in code_str: return 'urine'
  if '(ur)' in name_str: return 'urine'
  if 'mict' in name_str: return 'urine'
  if 'lcr' in name_str: return 'cerebrospinal_fluid'
  if 'l.périt' in name_str: return 'peritoneal_fluid'
  if 'l.asc' in name_str: return 'peritoneal_fluid'
  if 'l.ple' in name_str: return 'pleural_fluid'
  if 'lba' in name_str: return 'bronchoalveolar_lavage_fluid'
  if 'l.bio' in name_str: return 'unspecified_fluid'
  if 'liq' in name_str: return 'unspecified_fluid'
  if 'autres' in code_str: return 'other'
  else:
    #print('Unrecognized sample site: ', name_str, code_str)
    return 'blood'

def map_lab_result_value(result_string):
  if result_string is None: return ''
  result_string = str(result_string) \
    .replace('<', '') \
    .replace('>', '') \
    .replace(',', '.') \
    .strip()
  if result_string == '':
    return ''
  else:
    return float(result_string)

def map_pcr_name(name):
  name_str = str(name).lower().strip()
  if 'chlamydia' in name_str: return 'chlamydia'
  if 'gonocoque' in name_str: return 'gonorrhea'
  if 'l. monocytogenes' in name_str: return 'listeria_monocytogenes'
  if 'covid' in name_str: return 'sars_cov2'
  if 'herpès' in name_str: return 'herpes_simplex_virus'
  if 'influenza a' in name_str: return 'influenza_a'
  if 'influenza b' in name_str: return 'influenza_b'
  if 'rsv' in name_str: return 'respiratory_syncytial_virus'
  if 'virus jc' in name_str: return 'jc_virus'
  if 'vbk' in name_str: return 'bk_virus'
  if 'vzv' in name_str: return 'varicella_zoster_virus'
  print('Invalid PCR name: %s' % name_str)

def map_pcr_result_value(result):
  result_str = str(result).strip().lower()
  if 'positif' in result_str:  return 'positive'
  if 'négatif' in result_str: return 'negative'
  if 'non détecté' in result_str: return 'negative'
  if result_str == 'test annulé': return ''
  if result_str == 'annulé': return ''
  if result_str == 'non valide': return ''
  if result_str == 'en attente': return ''
  if result_str == 'rapp. numérisé': return ''
  #print('Invalid PCR result value: %s' % result_str)
  return ''

def map_pcr_result_status(result):
  if result is None: return ''
  result_str = str(result).lower().strip()
  if result_str == 'test annulé': return 'cancelled'
  if result_str == 'annulé': return 'cancelled'
  if result_str == 'non valide': return 'cancelled'
  if result_str == 'en attente': return 'pending'
  if result_str == 'rapp. numérisé': return 'resulted'
  if 'positif' in result_str: return 'resulted'
  if 'négatif' in result_str: return 'resulted'
  if 'détecté' in result_str: return 'resulted'
  print('Invalid PCR result status: %s' % result_str)
  return ''

def map_pcr_sample_site(name, code):
  name = (str(name).strip()).lower()
  code = (str(code).strip()).lower()
  
  if 'écouvillon' in code: return 'nasopharyngeal_swab'
  if 'urine' in code: return 'urine'
  if 'sang' in code: return 'blood'
  if ('rsv' in name or 'flu' in name) and 'micro' in code: 
    return 'nasopharyngeal_swab'
  if ('vzv pcr' in name or 'herpès pcr' in name or 'virus jc' in name or \
   'l. monocytogenes pcr' in name or 'vbk' in name ) and 'autres' in code:
    return 'cerebrospinal_fluid'
  if 'n.2019-ncov (covid-19)' in name and 'autres' in code:
    return 'nasopharyngeal_swab'
  else:
    print('Unrecognized sample site: %s, %s' % (code, name))
    return ''

def map_culture_type(culture_type):
  type_str = str(culture_type).strip().lower()
  if 'myco/f lytic' in type_str: return 'myco_culture'
  elif 'culture virale' in type_str: return 'viral_culture'
  else: return 'bacterial_culture'

def map_culture_specimen_type(type_desc, site_desc):

  type_desc = (str(type_desc).strip()).lower()
  site_desc = (str(site_desc).strip()).lower()

  desc = type_desc + ' ' + site_desc
  if 'hémoculture' in desc: return 'blood'
  if 'expectoration' in desc: return 'expectorated_sputum'
  if 'moelle' in desc: return 'bone_marrow'
  if 'prothèse' in desc: return 'prosthesis'
  if 'selles' in desc: return 'stool'
  if 'erv par culture' in desc: return 'rectal_swab'
  if 'sarm par culture' in desc: return 'nasal_swab'
  if 'gorge' in desc: return 'nasal_swab'
  if 'sécrétions nasales' in desc: return 'nasal_secretions'
  if 'céphalo-rachidien' in desc: return 'cerebrospinal_fluid'
  if 'biopsie d\'os' in desc: return 'bone_biopsy'
  if 'biopsie pulmonaire' in desc: return 'lung_biopsy'
  if 'biopsie pulmonaire' in desc: return 'lung_biopsy'
  if 'biopsie cutanée' in desc: return 'skin_biopsy'
  if 'biopsie de ganglion cutané' in desc: return 'lymph_node_biopsy'
  if 'corps étransfer' in desc: return 'foreign_body'
  if 'intravasculaire' in desc: return 'intravascular_catheter'
  if 'liq préservation' in desc: return 'preservation_liquid'
  if 'ascite' in desc: return 'ascites_fluid'
  if 'pleural' in desc: return 'pleural_fluid'
  if 'abcès du cerveau' in desc: return 'cerebral_abscess'
  if 'pus superficiel' in desc: return 'superficial_pus'
  if 'pus profond' in desc: return 'deep_pus'
  if 'urine' in desc: return 'urine'
  if 'bronchique' in desc: return 'bronchial_secretions'
  if 'sang' in desc: return 'blood'
  if 'urine' in desc: return 'urine'
  if 'lavage broncho-alvéolaire' in desc: return 'bronchoalveolar_lavage'
  if 'bronchique' in desc: return 'bronchial_secretions'
  if 'cimen micro' in desc: return 'other'
  if 'autres' in site_desc: return 'other'
  else:
    print('Unrecognized sample site: ' + desc)
    exit()

def map_culture_result_status(value):
  if value is None: return 'pending'
  value_str = str(value).strip().lower()
  if value_str == 'pos': return 'resulted'
  elif value_str == 'neg': return 'resulted'
  else: 
    print('Unrecognized culture status')
    return ''

def map_culture_growth_value(value):
  if value is None: return ''
  value_str = str(value).strip().lower()
  if value_str == 'pos': return 'positive'
  elif value_str == 'neg': return 'negative'
  else: 
    print('Unrecognized growth result')
    return ''

def map_episode_unit_type(unit_code, start_time=None):
  unit_code_str = str(unit_code)
   
  if start_time is not None:
    time_delta = get_hours_between_datetimes(
      '2020-01-01 00:00:00', start_time)
    
  if '10S' in unit_code_str or '10N' in unit_code_str:
    return 'intensive_care'
  if '13SM' in unit_code_str:
    return 'high_dependency'
  if '8NC' in unit_code_str:
    if time_delta > 0:
      return 'intensive_care'
    else:
      return 'coronary_care'

  if 'ER' in unit_code_str:
    return 'emergency_room'
  
  return 'inpatient_ward'

def map_observation_name(name):
  name_str = str(name).lower().strip()
  if name_str == 'fio2': return 'fraction_inspired_oxygen'
  if name_str == 'sat o2 art': return 'oxygen_saturation'
  if name_str == 'sat_o2': return 'oxygen_saturation'
  if name_str == 'température': return 'temperature'
  if name_str == 'tension_systol': return 'systolic_blood_pressure'
  if name_str == 'tension_diastol': return 'diastolic_blood_pressure'
  if name_str == 'rythme_resp': return 'respiratory_rate'
  if name_str == 'pouls': return 'heart_rate'
  if name_str == 'temp': return 'temperature'
  raise Exception('Unrecognized observation: %s' % name_str)

def map_drug_name(name):
  name_str = str(name).strip().lower()
  name_str = transliterate_string(name_str)
  name_str = name_str.replace('/ ', '/')
  name_str = name_str.replace(' /', '/')
  name_str = name_str.replace(' +', '+')
  name_str = name_str.replace('+ ', '+')
  if name_str in DRUG_NAME_MAP:
    return DRUG_NAME_MAP[name_str]

  if '*' in name_str or ',' in name_str or '(' in name_str:
    print('Unrecognized drug name: %s' % name_str)

  return name_str
  
def map_drug_route(route_code):

  route_code_str = str(route_code)

  if 'routecd' in route_code_str:
    route_code_str = route_code_str.replace('\n', ' ')
    route_code_parts = route_code_str.split('routecd')
    route_code_parts = [d.strip() for d in route_code_parts]
    route_code_parts = [d for d in route_code_parts if d != '']
    route_code_str = route_code_parts[0].replace('CY', '')

  route_code_str = route_code_str.lower().strip()
  
  if route_code_str in DRUG_ROUTE_MAP:
    return DRUG_ROUTE_MAP[route_code_str]
  else: 
    print('Invalid drug route: %s' % route_code_str)

def map_drug_frequency(frequency_code):

  #if frequency_code is None: return ''

  cd = str(frequency_code).lower().strip()
  
  if 'die' in cd or 'hs' in cd: return 'die'
  elif 'bid' in cd: return 'bid'
  elif 'tid' in cd: return 'tid'
  elif 'qid' in cd: return 'qid'
  elif 'q1j' in cd: return 'q1j'
  elif 'q2j' in cd: return 'q2j'
  elif 'q3j' in cd: return 'q3j'
  elif 'q5min' in cd: return 'q5min'
  elif 'q10min' in cd: return 'q10min'
  elif 'q15min' in cd: return 'q15min'
  elif 'q1-2' in cd: return 'q1-2h'
  elif 'q1-4' in cd: return 'q1-4h'
  elif 'q2-3' in cd: return 'q2-3h'
  elif 'q4-6' in cd: return 'q4-6h'
  elif 'q2' in cd: return 'q2h'
  elif 'q3' in cd: return 'q3h'
  elif 'q4' in cd: return 'q4h'
  elif 'q5' in cd: return 'q5h'
  elif 'q6' in cd: return 'q6h'
  elif 'q7' in cd: return 'q7h'
  elif 'q8' in cd: return 'q8h'
  elif 'q9' in cd: return 'q9h'
  elif 'q10' in cd: return 'q10h'
  elif 'q12' in cd: return 'q12h'
  elif 'q13' in cd: return 'q13h'
  elif 'q14' in cd: return 'q14h'
  elif 'q15' in cd: return 'q15h'
  elif 'q16' in cd: return 'q16h'
  elif 'q17' in cd: return 'q17h'
  elif 'q19' in cd: return 'q19h'
  elif 'q24' in cd: return 'q24h'
  elif 'qmois' in cd: return '1x monthly'
  elif cd in ['induc', 'induc0', '1dose']: return '1x'
  elif 'appel' in cd: return '1x'
  elif 'insam' in cd: return 'die'
  elif 'insmi' in cd: return 'die'
  elif 'insampm' in cd: return 'bid'
  elif 'ins820' in cd: return 'bid'
  elif 'inspm' in cd: return 'die'
  elif 'instid' in cd: return 'tid'
  elif 'qh' in cd: return 'die'
  elif 'pnuit' in cd: return 'die'
  elif 'perf' in cd: return 'perf'
  elif 'vaccin' in cd: return '1x'
  elif 'cejour' in cd: return '1x'
  elif cd in ['prechim', 'culots', 'gluco1/2', 'postculo', \
    'postdial', 'cvvhprot', 'directiv', 'test', 'prot', \
    'selh', 'selc', 'tqp', 'tqpa', 'tqpp', 'protp', \
    'acpiv', 'epidb']:
    return 'cond' # according to a protocol
  elif '1-2fpjp' in cd: return 'die-bid'
  elif '2-3fpjp' in cd: return 'bid-tid'
  elif '2-4fpjp' in cd: return 'bid-qid'
  elif '3-4fpjp' in cd: return 'tid-qid'
  elif cd in ['6fj2-21' '6fj6-21' '6fj8-22', '6fj6-21', \
    '6fj2-21', '6fj8-22']:
    return '6x daily'
  elif cd in ['l10', 'l19', 'l21', 'l6', 'l7', 'l9']: 
    return '1x weekly'
  elif cd in ['m10', 'm17', 'm21', 'm6', 'm7', 'm9']:
    return '1x weekly'
  elif cd in ['me10', 'me21', 'me6', 'me7', 'me9']:
    return '1x weekly'
  elif cd in ['j10', 'j21', 'j6', 'j7', 'j9']:
    return '1x weekly'
  elif cd in ['v10', 'v21', 'v6', 'v7', 'v9']: 
    return '1x weekly'
  elif cd in ['s10', 's21', 's6', 's7', 's9']:
    return '1x weekly'
  elif cd in ['d10', 'd12', 'd21', 'd6', 'd7', 'd9']:
    return '1x weekly'
  elif cd in ['lj10', 'lj21', 'lj9', 'lme21', 'lme9', \
    'lv21', 'lv9', 'mes9', 'mev9', 'mj21', 'mj9', \
    'ms9', 'mv10', 'mv9', 'mme9', 'dme9', 'ds19']:
    return '2x weekly'
  elif cd in ['lmev10', 'lmev12', 'lmev17', 'lmev21', \
    'lmv9', 'mjs17', 'mjs21', 'mjs9', 'lmev9', 'mmej9', 'dvs9']:
    return '3x weekly'
  elif cd in ['lmjs21', 'lmjvs9', 'lmmej9', 'dmjs19', \
    'dmjs21', 'dmjs8', 'dlmev21', 'dlmev9']:
    return '4x weekly'
  elif cd in ['lmmejv10', 'lmmejv9', 'lmmevs9', 'lmejvs9', \
    'dljvs9']:
    return '5x weekly'
  elif cd in ['6fs-d9', '6fs-j21', '6fs-l9', '6fs-m9', \
    '6fs-me9', '6fs-s9', '6fs-v21']:
    return '6x weekly'
  elif cd in ['p', 'prn']:
    return '' # PRN, unspecified frequency
  else: 
    return None
    print('Invalid drug frequency: %s' % cd)

def map_diagnosis_type(type):
  type_str = str(type).lower().strip()
  if 'diagnostic principal' in type_str:
    return 'principal'
  else: return 'secondary'
