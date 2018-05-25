"""
Saves info on the graph and the operations to visulize them via tensorboard
Some names need to be added so tensorboard can make sense of the information
"""

# import data and libraries
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
# add names to group operations together on the graph
with tf.name_scope("conv1"):
    layer_1 = tf.reshape(X, shape=[-1,28,28,1])
    # create a summary that will be logged to tensorboard (image)
    tf.summary.image('x', layer_1)
    layer_1 = tf.layers.conv2d(layer_1, 32, 5, activation=tf.nn.relu, name='conv2d')
    layer_1 = tf.layers.max_pooling2d(layer_1, 2, 2, name='max_pool')

# add names
with tf.name_scope("dense1"):
    layer_2 = tf.layers.flatten(layer_1, name='flatten')
    layer_2 = tf.layers.dense(layer_2, n_hidden_2, name='dense')
    logits = tf.layers.dense(layer_2, n_classes, name='logits')
    
learning_rate = 0.001
# add names
with tf.name_scope("train"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y) 
    average_cross_entropy = tf.reduce_mean(cross_entropy)
    # create a summary that will be logged to tensorboard (scalar)
    tf.summary.scalar('average_cross_entropy', average_cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_batch = optimizer.minimize(average_cross_entropy)

with tf.name_scope("init"):
    init_op = tf.global_variables_initializer()

# add names
with tf.name_scope("predictions"):
    predictions = tf.argmax(tf.nn.softmax(logits),1)
    correct_prediction_count = tf.equal( predictions, tf.argmax(Y, 1))
    average_accuracy = tf.reduce_mean(tf.cast(correct_prediction_count, tf.float32))
    # creates a summary that will be logged to tensorboard
    tf.summary.scalar('average_accuracy', average_accuracy)


# creates a summary with all the summaries' information
merged_summary = tf.summary.merge_all()

#same as before
n_steps = 300
batch_size = 128
display_step = 100
tensorboard_step = 10


with tf.Session() as sess:
    sess.run(init_op)
    print ('Model initialised')
    # write the graph info to the tmp folder
    # tensorboard can now be accessed via the following command
    # tensorboard --logdir tmp/tensorboard_data
    writer = tf.summary.FileWriter('tmp/tensorboard_data')
    writer.add_graph(sess.graph)
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

        # print info to tensorboard log every tensorboard_step
        if step % tensorboard_step == 0:
            summary = sess.run(merged_summary, feed_dict={X: batch_x, Y: batch_y})
            writer.add_summary(summary, step)

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