import csv

def read_csv(file_name):
  data_rows = []
  with open(file_name) as data_file:
    reader = csv.reader(data_file)
    row_count = 0
    for row in reader:
      if not row_count == 0:
        data_rows.append(row)
      row_count += 1

  return data_rows

def write_csv(headers, data, file_name):

  with open(file_name, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
    writer.writerow(headers)

    for row in data:
      writer.writerow(row)
      