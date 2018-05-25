"""
A simple neural network
Powered by Tensorflow
Trains to recognize the MNIST digits

"""

# import data and libraries
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# define placeholders to hold the train/test data
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# define variables for weights/biases to be updated in the training
n_hidden_1 = 256 
n_hidden_2 = 256 
weight_1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
weight_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
weight_3 = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
bias_1 = tf.Variable(tf.random_normal([n_hidden_1]))
bias_2 = tf.Variable(tf.random_normal([n_hidden_2]))
bias_3 = tf.Variable(tf.random_normal([n_classes]))

# define how network uses weights and biases
layer_1 = tf.add(tf.matmul(X, weight_1), bias_1)
layer_2 = tf.add(tf.matmul(layer_1, weight_2), bias_2)
logits = tf.add(tf.matmul(layer_2, weight_3), bias_3)

# define train operation to minimize average batch cross entropy 
learning_rate = 0.1
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y) # returns a 1-D Tensor of length batch_size
average_cross_entropy = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_batch = optimizer.minimize(average_cross_entropy)

# define init operation to randomly assign values to weights/biases 
init_op = tf.global_variables_initializer()

# run the init, train, and test operations
# until then, the operation were defined but nothing was run
# within the tensorflow session, all the operations needed for the
# init_op, train_batch, and average_cross_entropy will happen
n_steps = 1000
batch_size = 128
with tf.Session() as sess:
    sess.run(init_op)
    print ('Model initialised')
    for step in range(0, n_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_batch, feed_dict={X: batch_x, Y: batch_y})
        # additional information every display step
    print ('Model trained')
    average_cross_entropy = sess.run(
            average_cross_entropy, feed_dict={X: mnist.test.images,
                                              Y: mnist.test.labels})
    print ('Test average_cross_entropy: ', average_cross_entropy)