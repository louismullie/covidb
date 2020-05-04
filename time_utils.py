from datetime import datetime

def get_current_datetime():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_datetime_seconds(dt, dt_format="%Y-%m-%d %H:%M:%S"):
  return int(datetime.strptime(str(dt), dt_format).strftime('%s'))

def get_hours_between_datetimes(dt_1, dt_2, default_now = False, 
    dt1_format="%Y-%m-%d %H:%M:%S", dt2_format="%Y-%m-%d %H:%M:%S"):
  
  if (dt_2 is None or dt_2 == '' or \
    dt_2 == 'NaT' or dt_2 == 'None') and default_now:
    dt_2 = get_current_datetime()

  dt_1 = str(dt_1).split('.')[0]
  dt_2 = str(dt_2).split('.')[0]
  d1 = datetime.strptime(dt_1, dt1_format)
  d2 = datetime.strptime(dt_2, dt2_format)
  
  td = d2 - d1
  hours = td.total_seconds() / 60 / 60
  return int(hours)