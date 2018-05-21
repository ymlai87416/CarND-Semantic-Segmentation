import tensorflow as tf
import os.path

from tensorflow.python.platform import gfile
with tf.Session() as sess:
    vgg_path =os.path.join('.\\data', 'vgg')
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(
        sess,
        [vgg_tag],
        vgg_path)
LOGDIR=os.path.join('.\\data', 'vgg_log')
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)