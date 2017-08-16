import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

not_mnist = input_data.read_data_sets("notMNIST_data/", one_hot=True)

# your code here.
x = tf.placeholder(tf.float32, [None, 784]) # dimensionality of a single flattened 28 by 28 pixel MNIST image
y_ = tf.placeholder(tf.float32, [None, 10]) # target output classes

W = tf.Variable(tf.zeros([784, 10])) # 784x10 matrix (input feature =784, output = 10)
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b #linear regression

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) #softmax + mean square error
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #gradient descent

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) #assign initial values to each variables


for _ in range(1000):
    batch = not_mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # argmax +  check if our prediction matches the truth
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

print("test accuracy %g"%accuracy.eval(feed_dict={x: not_mnist.test.images, y_: not_mnist.test.labels})) #evaluate accuracy

