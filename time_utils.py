from datetime import datetime

def get_datetime_seconds(dt, dt_format="%Y-%m-%d %H:%M:%S"):
  return int(datetime.strptime(str(dt), dt_format).strftime('%s'))

def get_hours_between_datetimes(dt_1, dt_2, dt1_format="%Y-%m-%d %H:%M:%S", dt2_format="%Y-%m-%d %H:%M:%S"):
  d1 = datetime.strptime(str(dt_1), dt1_format)
  d2 = datetime.strptime(str(dt_2), dt2_format)
  td = d1 - d2
  hours = abs(td.total_seconds() / 60 / 60)
  return hours