import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
sess = tf.InteractiveSession()


def build_graph():
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = slim.conv2d(x_image, 32, [3, 3], 1, padding='SAME')
    h_pool1 = slim.max_pool2d(h_conv1, [2, 2], [2, 2], padding='SAME')

    h_conv2 = slim.conv2d(h_pool1, 64, [3, 3], 1, padding='SAME')
    h_pool2 = slim.max_pool2d(h_conv2, [2, 2], [2, 2], padding='SAME')

    h_conv3 = slim.conv2d(h_pool2, 128, [3, 3], 1, padding='SAME')
    h_pool3 = slim.max_pool2d(h_conv3, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(h_pool3)
    h_fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 256, activation_fn=tf.nn.relu)
    y_conv = slim.fully_connected(slim.dropout(h_fc1, keep_prob), 10, activation_fn=None)

    predict = tf.argmax(y_conv, 1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {
        'x': x,
        'y_': y_,
        'keep_prob': keep_prob,
        'accuracy': accuracy,
        'train_step': train_step,
        'y_conv': y_conv,
        'predict': predict,
    }


if __name__ == '__main__':
    graph = build_graph()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(1001):
        batch = mnist.train.next_batch(100)
        sess.run(graph['train_step'], feed_dict={graph['x']: batch[0], graph['y_']: batch[1], graph['keep_prob']: 0.6})

        if i % 100 == 0:
            train_accuracy = sess.run(graph['accuracy'], feed_dict={graph['x']: batch[0], graph['y_']: batch[1], graph['keep_prob']: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 1000 == 0:
            saver.save(sess, './model/model.ckpt')
