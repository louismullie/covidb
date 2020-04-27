import pandas as pd
import numpy as np
import re

COLUMN_WIDTH = 40

def ellipsize(string, length):
  if len(string) < length:
    return string
  else: 
    first_half = string[0:int(length/2-2)]
    second_half = string[-int(length/2-2):]
    return first_half + '(..)' + second_half

def most_frequent(data):
  return max(set(data), key=data.count)

def least_frequent(data):
  return min(set(data), key=data.count)

def pad_lines(value, length):
  str_value = str(value)
  for i in range(0, length-len(value)):
    str_value += '-'
  return str_value

def pad_height(lines, height):
  for i in range(0, height-len(lines)):
    lines.append(pad_whitespace('', COLUMN_WIDTH))
  return lines

def pad_whitespace(value, length):
  str_value = str(value)
  for i in range(0, length-len(value)):
    str_value += ' '
  return str_value

def tabulate_columns(column_names, rows, offset=0):

  padded_column_names = [
    pad_whitespace(cn, COLUMN_WIDTH) for cn in column_names
  ]
  padded_lines = [
    pad_lines('', COLUMN_WIDTH) for cn in column_names
  ]
  table_header = '| ' + ' | '.join(padded_column_names) + ' |'
  table_separator = '+-' + '-+-'.join(padded_lines) + '-+'

  text_lines = [table_separator, table_header, table_separator]

  column_index = offset
  table_lines = []

  for column_name in column_names:
    lines = tabulate_column(column_name, rows, column_index)
    lines = pad_height(lines, 10)
    line_index = 0
    for line in lines:
      if line_index >= len(table_lines):
        table_lines.append([])
      table_lines[line_index].append(line)
      line_index += 1
    column_index += 1

  for line in table_lines:
    text_lines.append('| ' + ' | '.join(line) + ' |')

  text_lines.append(table_separator)

  column_index = offset
  table_lines = []

  for column_name in column_names:
    lines = analyze_column(column_name, rows, column_index)
    lines = pad_height(lines, 10)
    line_index = 0
    for line in lines:
      if line_index >= len(table_lines):
        table_lines.append([])
      table_lines[line_index].append(line)
      line_index += 1
    column_index += 1

  for line in table_lines:
    text_lines.append('| ' + ' | '.join(line) + ' |')

  text_lines.append(table_separator)
  text_lines = [l for l in text_lines if \
    l.replace(' ', '').replace('|', '') != '']
  print('\n'.join(text_lines))

def tabulate_column(col_name, rows, col_index, default='missing data'):
  values = []
  for row in rows:
    if row[col_index] == '':
      values.append(default)
    else:
      values.append(row[col_index])
  values = np.asarray(values)
  num_values = values.shape[0]

  unique_values = np.unique(values)
  table = []
  for unique_value in unique_values:
    count = np.count_nonzero(values == unique_value)
    perc = float(count) / num_values
    table.append([unique_value, count, perc])
  table.sort(key=lambda x: -x[1])
  lines = []
  i = 0
  for row in table:
    if i == 5 and i != len(table) - 1:
      lines.append(pad_whitespace('...', COLUMN_WIDTH))
      i += 1
      continue
    if i > 5 and i != len(table) - 1:
      i += 1
      continue
    perc = str(round(row[2] * 100, 1)) + '%'
    val = ellipsize(str(row[0]), 20)
    txt = val + ': n = ' + \
        str(row[1]) + ' (' + perc + ')'
    txt = pad_whitespace(txt, COLUMN_WIDTH)
    lines.append(txt)
    i += 1

  return lines

def analyze_column(col_name, rows, col_index):

  col_data = []
  for row in rows:
    col_data.append(row[col_index])

  null_col_data = [1 if x == None else 0 for x in col_data]
  empty_col_data = [1 if str(x) == '' else 0 for x in col_data]
  
  lines = []

  PATTERNS = [
    ['positive integer', re.compile('^[0-9]+$')],
    ['positive float', re.compile('^[0-9]+\.[0-9]+$')],
    ['letters only', re.compile('^[A-Za-z]+$')],
    ['letters and numbers', re.compile('^[0-9A-Za-z]+$')],
    ['datetime', re.compile('^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$')],
    ['free string', re.compile('^.+$')],
  ]
  no_outside = 0
  validation_pattern = None

  empty_els = np.sum([1 for el in col_data if el == ''])

  for pattern_name, pattern_re in PATTERNS:
    matching_els = [1 for el in col_data if pattern_re.match(str(el))]
    no_outside = len(col_data) - len(matching_els) - empty_els

    if (len(matching_els) + empty_els) >= len(col_data) * 0.9 and no_outside < 100:
      lines.append('validation charset: %s' % pattern_name)
      validation_pattern = pattern_name
      break
  
  lines.append('total no. of values: %d' % len(col_data))
  lines.append('no. illegal values: %d' % no_outside)
  if no_outside == 40:
    for el in col_data:
      if el != '' and not re.compile('^[0-9]+\.[0-9]+$').match(el):
        print(el)
        exit()
  lines.append('no. empty values: %d' % np.count_nonzero(empty_col_data))

  # Sample the first 50 to avoid computationally intensive calculations
  uniq_col_data = np.unique(col_data[0:50])

  if len(col_data) != 0 and (float(len(uniq_col_data)) / len(col_data)) < 0.75:
    
    least_freq = least_frequent(col_data)
    most_freq = most_frequent(col_data)
    uniq_col_data = np.unique(col_data)

    mf_col_data = [1 if x == most_freq else 0 for x in col_data]
    lf_col_data = [1 if x == least_freq else 0 for x in col_data]

    lines.append('no. of unique values: %d' % len(uniq_col_data))
    lines.append('no. with most frequent value: %s' % np.count_nonzero(mf_col_data))
    lines.append('no. with least frequent value: %s' % np.count_nonzero(lf_col_data))
  
  if validation_pattern in ['positive float']:
    col_data_num = [float(x) for x in col_data if x != '']
    mean = round(np.mean(col_data_num), 2)
    stdev = round(np.std(col_data_num), 2)
    lines.append('mean value (stdev): %f +/- %f' % (mean, stdev))

  return [pad_whitespace(line, COLUMN_WIDTH) for line in lines]