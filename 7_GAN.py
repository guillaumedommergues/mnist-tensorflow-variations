"""
Just for fun: a completely different architecture
to create fake digits as opposed to recognizing them
Inspired by a deep learning class from Udacity

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def model_inputs(real_dim, z_dim):
    """
    create input placeholders
    real_dim: dimension of the images we'll try to generate
    z_dim: dimension of the noise we'll start with to generate images

    """
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')    
    return inputs_real, inputs_z

def generator(z, out_dim, n_units=128, reuse=False):
    """
    architecture of neural network used to generate images from z, ie
    some randomly generated noise

    """
    with tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.nn.relu(h1)
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)
        return out

def discriminator(x, n_units=128, reuse=False):
    """
    architecture of neural network used to find which images are real
    and which are fake

    """
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.nn.relu(h1)        
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)
        return out, logits

# Create inputs
input_size = 784
z_size = 100
tf.reset_default_graph()
input_real, input_z = model_inputs(input_size, z_size)

# Create models
g_hidden_size = 128
d_hidden_size = 128
g_model = generator(input_z, input_size, n_units=g_hidden_size)
d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, n_units=d_hidden_size)

# Calculate losses
# loss of discriminator increases when it wrongly categorizes either true or fake input
d_loss_real = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                          labels=tf.ones_like(d_logits_real)))
d_loss_fake = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                          labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                     labels=tf.ones_like(d_logits_fake)))

# Define optimizers
learning_rate = 0.002
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

# Run model
batch_size = 100
n_epoch = 5
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        for step in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            # reshape and rescale images for the discriminator
            batch_images = batch[0].reshape((batch_size, 784))
            
            # creates random noise for the generator
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            
            # Run optimizers
            sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
        print("Step " + str(step))
        print("Discriminator Loss: " + "{:.2f}".format(train_loss_d))
        print("Generator Loss: " + "{:.2f}".format(train_loss_g))
    print ("model trained") 
    sample_z = np.random.uniform(-1, 1, size=(16, z_size))
    gen_samples = sess.run(
                   generator(input_z, input_size, n_units=g_hidden_size, reuse=True),#, alpha=alpha),
                   feed_dict={input_z: sample_z})

i = 1
for image in gen_samples[0:10]:
    image = image.reshape(28,28)
    sub = plt.subplot(5,2,i)
    sub.imshow(image)
    i = i + 1
plt.show()



