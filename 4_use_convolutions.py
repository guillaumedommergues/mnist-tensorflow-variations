"""
Use convolution, a way to keep information
on how the different pixels are placed next to one another
as opposed to flattening all the pixels in one long array

"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 784 
n_classes = 10 
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

n_hidden_1 = 256 
n_hidden_2 = 256 
# reshape the input in the format [batch_size, width, height, channels]
layer_1 = tf.reshape(X, shape=[-1,28,28,1])
# use convolutions and pooling
layer_1 = tf.layers.conv2d(layer_1, 32, 5, activation=tf.nn.relu)
layer_1 = tf.layers.max_pooling2d(layer_1, 2, 2)
# flatten layer and finish model with standard dense layers
layer_2 = tf.layers.flatten(layer_1)
layer_2 = tf.layers.dense(layer_2, n_hidden_2)
logits = tf.layers.dense(layer_2, n_classes) 

# define the rest of the graph, same as before
# the learning rate needs to be smaller for better results
learning_rate = 0.001
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y) 
average_cross_entropy = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_batch = optimizer.minimize(average_cross_entropy)

init_op = tf.global_variables_initializer()

predictions = tf.argmax(tf.nn.softmax(logits),1)
correct_prediction_count = tf.equal( predictions, tf.argmax(Y, 1))
average_accuracy = tf.reduce_mean(tf.cast(correct_prediction_count, tf.float32))

# run the operation, same as before
n_steps = 1000
batch_size = 128
display_step = 100
with tf.Session() as sess:
    sess.run(init_op)
    print ('Model initialised')
    for step in range(0, n_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_batch, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run(
                    [average_cross_entropy, average_accuracy],
                    feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step))
            print("Average cross entropy: " + "{:.2f}".format(loss))
            print("Training accuracy: " + "{:.2f}".format(acc))

    predictions = sess.run(
            predictions, feed_dict={X: mnist.test.images,
                                    Y:mnist.test.labels})
    test_accuracy = sess.run(
            average_accuracy, feed_dict={X: mnist.test.images,
                                    Y:mnist.test.labels})
    print("Average test accuracy after training: " + "{:.2f}".format(test_accuracy))


n_images = 10
plt.figure(figsize=(3,n_images*3))
for i in range(1,n_images+1):
    plt.subplot(n_images/2,2,i)
    plt.title('Prediction: '+ str(predictions[i]))
    plt.imshow(mnist.test.images[i].reshape([28,28]))
plt.tight_layout()  
plt.show()   