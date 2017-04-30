from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])  # input, pixels of the image flattened into a vector
W = tf.Variable(tf.zeros([784, 10]))  # weights (like the coefficients in an svm)
b = tf.Variable(tf.zeros([10]))  # biases, kind of force the prediction one way?

y = tf.nn.softmax(tf.matmul(x, W) + b)  # calculated predicted probabilities for each category
y_ = tf.placeholder(tf.float32, [None, 10])  # the correct answers (for that image) used for training


# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # this is the equation but dont use because unstable.
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  ts, weights =  sess.run([train_step, W], feed_dict={x: batch_xs, y_: batch_ys})

print("weights len", len(weights))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))