import pandas as pd
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt 

def plot_compare_kde(var_name, comparator_name, label_pos, label_neg, values_pos, values_neg, min_value, max_value):

  ax = plt.gca()

  ax.set_title(var_name + ' according to ' + comparator_name, fontdict={'fontsize': 6})

  col_pos = '%s (n=%d samples)' % (label_pos, len(values_pos))
  col_neg = '%s (n=%d samples)' % (label_neg, len(values_neg))
  
  shuffle(values_pos)
  shuffle(values_neg)

  if len(values_pos) > len(values_neg):
    values_pos = values_pos[0:len(values_neg)-1]
  else:
    values_neg = values_neg[0:len(values_pos)-1]

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
  ax.tick_params(labelsize=6)
  ax.legend(loc=4, fontsize=5)