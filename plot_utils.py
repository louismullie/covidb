import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

def plot_compare_kde(var_name, comparator_name, label_pos, label_neg, values_pos, values_neg, min_value, max_value):

  ax = plt.gca()

  plt.title(var_name + ' according to ' + comparator_name)

  col_pos = '%s (n=%d samples)' % (label_pos, len(values_pos))
  col_neg = '%s (n=%d samples)' % (label_neg, len(values_neg))

  df_pos = pd.DataFrame(values_pos, columns=[col_pos])
  df_neg = pd.DataFrame(values_neg, columns=[col_neg])

  df_pos.plot(kind='density',color='red',ax=ax)
  df_neg.plot(kind='density',color='green',ax=ax)
  
  all_values = values_pos + values_neg

  if np.min(all_values) > min_value:
    min_value = np.min(all_values)

  if np.max(all_values) < max_value:
    max_value = np.max(all_values)

  ax.set_xlim(xmin=min_value, xmax=max_value)

  plt.show()