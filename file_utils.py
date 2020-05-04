import csv

def read_csv(file_name, remove_duplicates=False):

  data_rows = []
  lines_seen = set()
  duplicates_removed = 0

  with open(file_name) as data_file:
    reader = csv.reader(data_file)
    row_count = 0

    for row in reader:

      if row_count == 0: 
        row_count += 1
        continue

      if ''.join(row) in lines_seen:
        if remove_duplicates: 
          duplicates_removed += 1
          continue

      data_rows.append(row)
      lines_seen.add(''.join(row))

      row_count += 1
  
  if remove_duplicates:
    duplicates_msg = '- removed %d duplicates' % \
      duplicates_removed
  else: 
    duplicates_msg = ''

  print('Read %s %s' % (file_name, duplicates_msg))
  return data_rows

def write_csv(headers, data, file_name, remove_duplicates=False):
  
  with open(file_name, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
    writer.writerow(headers)

    lines_seen = set()
    duplicates_removed = 0

    for row in data:
      line = ''.join([str(x) for x in row])
      if line in lines_seen:
        if remove_duplicates: 
          duplicates_removed += 1
          continue
      writer.writerow(row)
      lines_seen.add(line)
  
  if remove_duplicates:
    duplicates_msg = '- removed %d duplicates' % \
      duplicates_removed
  else: 
    duplicates_msg = ''

  print('Wrote %s %s' % (file_name, duplicates_msg))