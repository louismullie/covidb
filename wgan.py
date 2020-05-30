'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm

from gan_utils import normalization, renormalization, rounding
from gan_utils import xavier_init
from gan_utils import binary_sampler, uniform_sampler, sample_batch_index


def gain (data_x, gain_parameters):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)  
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
  
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1) 
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    #D_h2 = tf.nn.dropout(D_h2, rate=0.3)
    D_prob = tf.nn.sigmoid()
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)

  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  X_true = M * X
  X_pred = M * G_sample
  
  MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  Hu_loss = tf.reduce_mean(tf.keras.losses.Huber()(X_true, X_pred))
  KL_loss = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(X_true, X_pred))
  
  D_loss = D_loss_temp
  alpha, beta, delta = 5, 0.05, 10 ### Extract
  G_loss = G_loss_temp + alpha * MSE_loss + beta * KL_loss  #.sqrt(MSE_loss)
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.1).minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.1).minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  losses = {'D': [], 'G': [], 'K-L': [], 'MSE': [], 'Hu': []}
  
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Get batch coordinates
    batch_idx = sample_batch_index(no, batch_size)

    # Get (normalized) data at coordinates
    X_mb = norm_data_x[batch_idx, :]  

    # Get auxiliary (missingness) matrix
    M_mb = data_m[batch_idx, :]  

    # Generate a random normal distribution (batch_size X dim)  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 

    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Mask * Data + (1- Mask) * Random
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr, KL_loss_curr, Hu_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss, KL_loss, Hu_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
    
    #if int(MSE_loss_curr * 1000) % 10 == 0:
    losses['D'].append(D_loss_curr )
    losses['G'].append(G_loss_curr )
    losses['K-L'].append(KL_loss_curr * beta)
    losses['MSE'].append(MSE_loss_curr * alpha)
    losses['Hu'].append(Hu_loss_curr * delta)
    print(it, G_loss_curr - MSE_loss_curr * alpha - KL_loss_curr * beta, MSE_loss_curr * alpha, KL_loss_curr * beta, G_loss_curr, MSE_loss_curr)
    
    if MSE_loss_curr < 0.01:
      break
  
  import matplotlib.pyplot as plt
  plt.plot(losses['D'], label='discriminator', lw=1)
  plt.plot(losses['G'], label='generator', lw=1)
  plt.plot(losses['K-L'], label='K-L', lw=1)
  plt.plot(losses['MSE'], label='MSE', lw=1)
  plt.plot(losses['Hu'], label='Huber', lw=1)
  plt.legend()
  plt.show()
  
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return imputed_data