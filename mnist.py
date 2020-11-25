from PIL import Image
import tensorflow as tf
from train import build_graph
global_times = 0


def image_prepare(file_path):
    im = Image.open(file_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    tv = list(im.getdata())
    return tv


def predict_prepare():
    sess = tf.Session()
    graph = build_graph()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('./model/')
    if ckpt:
        saver.restore(sess, ckpt)
    return graph, sess


def recognition(file_path):
    global global_times
    if global_times == 0:
        global graph, sess
        graph, sess = predict_prepare()
        image = image_prepare(file_path)
        predict = sess.run(graph['predict'], feed_dict={graph['x']: [image], graph['keep_prob']: 1.0})
        global_times = 1

        # print('recognize result:')
        # print(predict)
        return predict

    else:

        image = image_prepare(file_path)
        predict = sess.run(graph['predict'], feed_dict={graph['x']: [image], graph['keep_prob']: 1.0})

        # print('recognize result:')
        # print(predict)
        return predict



