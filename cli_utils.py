import pandas as pd
import numpy as np

def tabulate_column(col_name, rows, col_index, default='empty'):
  values = []
  for row in rows:
    if row[col_index] is None:
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
  print('\nValues for: %s' % col_name)
  i = 0
  for row in table:
    if i == 5 and i != len(table) - 1:
      print('  ...')
      i += 1
      continue
    if i > 5 and i != len(table) - 1:
      i += 1
      continue
    perc = str(round(row[2] * 100, 1)) + '%'
    print('  ' + str(row[0]) + ': n = ' + \
        str(row[1]) + ' (' + perc + ')')
    i += 1