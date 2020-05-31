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
  X_dim = 99
  z_dim = 30
  noise_factor = 0.25
  
  dropout = tf.placeholder(tf.int32, shape = [1])
  
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  z = tf.placeholder(tf.float32, shape=[None, z_dim])
  
  # Encoded vector
  X_e = tf.placeholder(tf.float32, shape=(None, dim))
  
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  """ Q(X|z) """
  Q_W1 = tf.Variable(xavier_init([dim, h_dim]))
  Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

  Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
  Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

  Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
  Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))
  
  """ P(X|z) """
  P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
  P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

  P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
  P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
  
  theta_E = [Q_W1, Q_b1, Q_W2_mu, Q_b2_mu, Q_W2_sigma, Q_b2_sigma, P_W1, P_b1, P_W2, P_b2]
  
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
  
  ## VAE functions
  def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar

  def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

  def sample_z(mu, log_var):
      eps = tf.random_normal(shape=tf.shape(mu))
      return mu + tf.exp(log_var / 2) * eps

  ## GAIN functions
  
  # Generator
  def generator(x,m, use_dropout):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1)
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    if use_dropout: G_h1 = tf.nn.dropout(G_h1, rate=0.3)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    if use_dropout: G_h2 = tf.nn.dropout(G_h2, rate=0.3)
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob
  
  # Discriminator
  def discriminator(x, h, use_dropout):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1) 
    if use_dropout: D_h1 = tf.nn.dropout(D_h1, rate=0.5)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    if use_dropout: D_h2 = tf.nn.dropout(D_h2, rate=0.5)
    D_prob = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3)
    return D_prob

  if tf.reduce_sum(dropout) == 1: 
    use_dropout = True
  else:
    use_dropout = False

  alpha, beta, delta = 10, 0.01, 0.01 ### Extract
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M, use_dropout)
  
  # Encoder
  X_noise = G_sample + noise_factor * tf.random_normal(tf.shape(G_sample))
  X_noise = tf.clip_by_value(X_noise, 0., 1.)

  z_mu, z_logvar = Q(X_noise)
  z_sample = sample_z(z_mu, z_logvar)
  X_e, logits = P(z_sample)

  X_samples, _ = P(z)
  # E[log P(X|z)]
  recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=G_sample), 1)
  # D_KL(Q(z|X_noise) || P(z|X)); calculate in closed form as both dist. are Gaussian
  kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
  # VAE loss
  E_loss_temp = tf.reduce_mean(recon_loss + kl_loss) * beta
  
  # Combine with observed data
  Hat_X = X_e * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H, use_dropout)

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
  G_loss = G_loss_temp + alpha * MSE_loss + delta * tf.math.abs(KL_loss)
  E_loss = E_loss_temp
  
  ## GAIN solver
  E_solver = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.5).minimize(E_loss, var_list=theta_E)
  D_solver = tf.train.AdamOptimizer(learning_rate=0.000001, beta1=0.5).minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer(learning_rate=0.00002, beta1=0.5).minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  losses = {'E': [], 'D': [], 'G': [], 'K-L': [], 'MSE': [], 'Hu': []}
    
  dropout = tf.constant(1)
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
    
    _, E_loss_curr = sess.run([E_solver, E_loss_temp], 
        feed_dict={M: M_mb, X: X_mb, H: H_mb })
        
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
         feed_dict = {M: M_mb, X: X_mb, H: H_mb  })
         
    _, G_loss_curr, MSE_loss_curr, KL_loss_curr, Hu_loss_curr = \
       sess.run([G_solver, G_loss_temp, MSE_loss, KL_loss, Hu_loss],
       feed_dict = {X: X_mb, M: M_mb, H: H_mb })
    
    losses['E'].append(E_loss_curr)
    losses['D'].append(D_loss_curr)
    losses['G'].append(G_loss_curr )
    losses['MSE'].append(MSE_loss_curr * alpha)
    
    print('Iteration: %d, encoder: %.3f, discriminator: %.3f, generator: %.3f, MSE: %.3f' % 
      (it, E_loss_curr, D_loss_curr, G_loss_curr, MSE_loss_curr))
    
    #if MSE_loss_curr < 0.005:
    #  break
    
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
    
  dropout = tf.constant(0)
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb })[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
  
  import matplotlib.pyplot as plt
  plt.plot(losses['E'], label='encoder', lw=1)
  plt.plot(losses['D'], label='discriminator', lw=1)
  plt.plot(losses['G'], label='generator', lw=1)
  #plt.plot(losses['K-L'], label='K-L', lw=1)
  plt.plot(losses['MSE'], label='MSE', lw=1)
  #plt.plot(losses['Hu'], label='Huber', lw=1)
  plt.legend()
  ax = plt.gca()
  ax.set_ylim([0.0, 1.0])
  plt.show()
  
  return imputed_data