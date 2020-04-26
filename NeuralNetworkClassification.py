#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# import matplotlib.pyplot as plt
# single_image = mnist.train.images[1].reshape(28, 28)
# plt.imshow(single_image, cmap="gist_gray")
# plt.show()

# PLACEHOLDERS (input data flattened & actual output data)
x_input = tf.placeholder(tf.float32, shape=[None, 784])
y_actual_output = tf.placeholder(tf.float32, shape=[None, 10])

# VARIABLES (weights & bias)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# CREATE GRAPH OPERATIONS
y_predicted_output = tf.matmul(x_input, W) + b

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_actual_output, logits=y_predicted_output))

# OPTIMIZER (for adjusting learning rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# CREATE SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x_input: batch_x, y_actual_output: batch_y})

    # EVALUATE THE MODEL
    correct_prediction = tf.equal(tf.argmax(y_predicted_output, 1), tf.argmax(y_actual_output, 1))

    # CONVERT [TRUE/FALSE] -> [1/0] AND FIND AVERAGE
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x_input: mnist.test.images, y_actual_output: mnist.test.labels}))
