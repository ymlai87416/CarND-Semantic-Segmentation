import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

import matplotlib as mpl
import matplotlib.pyplot as plt

gg = []

def debug_show_image(image):
    plt.ion()
    plt.figure()
    plt.imshow(image)
    plt.ioff()
    plt.show()

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(
        sess,
        [vgg_tag],
        vgg_path)

    detection_graph = tf.get_default_graph()
    vgg_input_tensor = detection_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = detection_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = detection_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = detection_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = detection_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    fcn_layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer7_conv_1x1')

    fcn_layer7_deconv = tf.layers.conv2d_transpose(fcn_layer7_conv_1x1, num_classes, 4, 2, padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer7_deconv')

    fcn_layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='SAME',
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer4_conv_1x1')

    intermediate_1 = tf.add(fcn_layer7_deconv, fcn_layer4_conv_1x1, name='intermediate_1')

    fcn_layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='SAME',
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer3_conv_1x1')

    intermediate_1_deconv = tf.layers.conv2d_transpose(intermediate_1, num_classes, 4, 2, padding='SAME',
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='intermediate_1_deconv')

    intermediate_2 = tf.add(intermediate_1_deconv, fcn_layer3_conv_1x1, name='intermediate_2')

    fcn_output = tf.layers.conv2d_transpose(intermediate_2, num_classes, 16, 8, padding='SAME',
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_output')

    return fcn_output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # freeze all convolution variables
    tvars = tf.trainable_variables()
    trainable_vars = [var for var in tvars if not(var.name.startswith('conv'))]

    print("Trainable parameters are: ")
    for var in trainable_vars:
        print(var.name + "\n")

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss, var_list=trainable_vars)

    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, writer=None, merged=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param writer: Tensorboard writer
    :param merged: Tensorboard merged summary
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())

    keep_prob_value = 0.5
    learning_rate_value = 0.001

    show_image = False

    step = 0
    for i in range(epochs):
        print("Epoch: = {:d}".format(i))
        for image, label in get_batches_fn(batch_size):
            #try:
            #    if not show_image:
            #        print(image[0].shape)
            #        #debug_show_image(image[0])
            #        plt.imsave("./data/training_sample.png", image[0])
            #        show_image = True
            #except:
            #    print("Oops! error")

            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: keep_prob_value, learning_rate: learning_rate_value})

            if step % 20 == 0 and writer is not None and merged is not None:
                result = sess.run(merged, feed_dict={input_image: image, correct_label: label, keep_prob: keep_prob_value, learning_rate: learning_rate_value})
                writer.add_summary(result, step)

            step = step+1

tests.test_train_nn(train_nn)

def make_train_video():
    pass

def make_test_video():
    pass

def make_real_video():
    pass


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 1
    batch_size = 16

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    LOGDIR = os.path.join('.\\data', 'fcn_log')

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], num_classes), name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, train_writer, merged)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
