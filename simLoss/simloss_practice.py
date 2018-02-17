import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None, name=None):
    add_layer.counter += 1
    with tf.name_scope("layer" + str(add_layer.counter)):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="Weights")
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="biases")
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        if name is None:
            pass
            # do nothing
        else:
            outputs = tf.identity(outputs, name)
    return outputs
add_layer.counter = 0

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1], name="xs")
ys = tf.placeholder(tf.float32, [None, 1], name="ys")

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu, name="layer1")
prediction = add_layer(l1, 10, 1, activation_function=None, name="prediction")

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]),name="loss")

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss,name="min_loss")

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("logs/", sess.graph)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
