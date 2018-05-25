"""
Add two ways to check model performance:
- a metric: the true positive ratio on the test set
- a visualization with matplotlib

"""

# import data and libraries
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# same as before 
n_input = 784 
n_classes = 10 
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

n_hidden_1 = 256 
n_hidden_2 = 256 
weight_1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
weight_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
weight_3 = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
bias_1 = tf.Variable(tf.random_normal([n_hidden_1]))
bias_2 = tf.Variable(tf.random_normal([n_hidden_2]))
bias_3 = tf.Variable(tf.random_normal([n_classes]))

layer_1 = tf.add(tf.matmul(X, weight_1), bias_1)
layer_2 = tf.add(tf.matmul(layer_1, weight_2), bias_2)
logits = tf.add(tf.matmul(layer_2, weight_3), bias_3)

learning_rate = 0.1
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y) 
average_cross_entropy = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_batch = optimizer.minimize(average_cross_entropy)

init_op = tf.global_variables_initializer()

# add predictions to know the model output on the test dataset
predictions = tf.argmax(tf.nn.softmax(logits),1)
# add average_accuracy operation to know ratio of correct answers
correct_prediction_count = tf.equal( predictions, tf.argmax(Y, 1))
average_accuracy = tf.reduce_mean(tf.cast(correct_prediction_count, tf.float32))


# run the operations defined
# now displaying cross entropy and accuracy on the training set every 100 steps
# and the accuracy on the test set after training 
n_steps = 1000
batch_size = 128
display_step = 100
with tf.Session() as sess:
    sess.run(init_op)
    print ('Model initialised')
    for step in range(0, n_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_batch, feed_dict={X: batch_x, Y: batch_y})
        # additional information every display step
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

# plot predictions next to actual images 
n_images = 10
plt.figure(figsize=(3,n_images*3))
for i in range(1,n_images+1):
    plt.subplot(n_images/2,2,i)
    plt.title('Prediction: '+ str(predictions[i]))
    plt.imshow(mnist.test.images[i].reshape([28,28]))
plt.tight_layout()  
plt.show()   