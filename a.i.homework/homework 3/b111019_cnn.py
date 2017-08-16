import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

not_mnist = input_data.read_data_sets("notMNIST_data/", one_hot=True)

# your code here.
x = tf.placeholder(tf.float32, shape = [None, 784]) #dimensionality of a single flattened 28 by 28 pixel MNIST image
y_ = tf.placeholder(tf.float32, shape = [None, 10]) # target output classes


def weight_variable(shape):  #weight
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape): #bias
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): #convolution layer
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x): #pooling layer
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # convolution layer(relu) 
h_pool1 = max_pool_2x2(h_conv1) # pooling layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #  reshape the tensor from the pooling layer into a batch of vectors 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) 

keep_prob = tf.placeholder(tf.float32) # probability that a neuron's output is kept during dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # handles scaling neuron outputs in addition to masking them

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #AdamOptimizer
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) # argmax + check if our prediction matches the truth
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

sess = tf.InteractiveSession()       
sess.run(tf.global_variables_initializer()) #assign initial values to each variables


for i in range(20000):
      batch = not_mnist.train.next_batch(50)
      if i%100 == 0: # add logging to every 100th iteration
          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
          print("step %d, training accuracy %g"%(i, train_accuracy))
          train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})  


print("test accuracy %g"%accuracy.eval(feed_dict={x:not_mnist.test.images, y_:not_mnist.test.labels, keep_prob:1.0})) #evaluate accuracy
                                
