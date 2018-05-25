"""
Simplify the graph by using the pre-built tensorflow layers
As opposed to re-creating the operations


"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 784 
n_classes = 10 
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# use the tf.layer.dense functions to simplify code
n_hidden_1 = 256 
n_hidden_2 = 256 
layer_1 = tf.layers.dense(X, n_hidden_1)
layer_2 = tf.layers.dense(layer_1, n_hidden_2)
logits = tf.layers.dense(layer_2, n_classes) 

# create the remaining of the graph, same as before
learning_rate = 0.1
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