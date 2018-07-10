import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import sys
import numpy as np
import utils

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
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer7_conv_1x1')

    fcn_layer7_deconv = tf.layers.conv2d_transpose(fcn_layer7_conv_1x1, num_classes, 4, 2, padding='SAME',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer7_deconv')

    vgg_layer4_out_scale = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scale')

    fcn_layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out_scale, num_classes, 1, padding='SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer4_conv_1x1')

    intermediate_1 = tf.add(fcn_layer7_deconv, fcn_layer4_conv_1x1, name='intermediate_1')

    vgg_layer3_out_scale = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scale')

    fcn_layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out_scale, num_classes, 1, padding='SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='fcn_layer3_conv_1x1')

    intermediate_1_deconv = tf.layers.conv2d_transpose(intermediate_1, num_classes, 4, 2, padding='SAME',
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='intermediate_1_deconv')

    intermediate_2 = tf.add(intermediate_1_deconv, fcn_layer3_conv_1x1, name='intermediate_2')

    fcn_output = tf.layers.conv2d_transpose(intermediate_2, num_classes, 16, 8, padding='SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
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

    #print("Trainable parameters are: ")
    #for var in trainable_vars:
    #    print(var.name + "\n")

    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")
    pred = tf.nn.softmax(logits)
    output = tf.identity(pred, 'prediction')

    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss, collections=['batch'])
    # add regularization to the loss
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('regularization loss', reg_losses, collections=['batch'])
    reg_constant = 0.01
    loss = cross_entropy_loss + reg_constant * reg_losses

    tf.summary.scalar('total loss', loss, collections=['batch'])

    prediction = tf.argmax(logits, 1)
    correct_label_flatten = tf.argmax(correct_label, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(prediction, correct_label_flatten), tf.float32))
    tf.summary.scalar('train_acc', acc, collections=['epoch_train'])
    tf.summary.scalar('validation_acc', acc, collections=['epoch_validate'])

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss, var_list=trainable_vars)

    return logits, training_operation, loss
tests.test_optimize(optimize)


def model_checkpoint_callback(sess, epoch, target):
    saver = tf.train.Saver()
    save_path = saver.save(sess, target)
    print("Model saved in path: %s" % save_path)

def log_validation_accuary(sess, epoch, writer, summary_validate, input_image, correct_label, keep_prob, gen_batch_func_validate, batch_size):
    sess.run(tf.local_variables_initializer())
    if summary_validate is not None:
        overall_result = {}
        total_cnt = 0

        for image, label in gen_batch_func_validate(batch_size):
            current_batch_size = len(label)
            total_cnt += current_batch_size
            valid_b = sess.run(summary_validate, feed_dict={input_image: image, correct_label: label, keep_prob: 1})

            valid_sum = tf.Summary()
            valid_sum.ParseFromString(valid_b)

            for i in valid_sum.value:
                overall_result[i.tag] = overall_result.get(i.tag, 0) + i.simple_value * 1.0 *current_batch_size

        for key in overall_result:
            overall_result[key] /= total_cnt

        summary = tf.Summary()

        for (key, value) in overall_result.items():
            summary.value.add(tag=key, simple_value=value)

        writer.add_summary(summary, epoch)

        return overall_result
    else:
        return {}

def log_batch_history(sess, writer, summary, batch_no):
    writer.add_summary(summary, batch_no)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, writer=None, get_batches_fn_validate=None, checkpoint_path=None):
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
    sess.run(tf.global_variables_initializer())

    keep_prob_value = 0.5
    learning_rate_value = 0.0001

    show_image = False

    summary_train = tf.summary.merge_all(key='epoch_train')
    summary_validate = tf.summary.merge_all(key='epoch_validate')
    summary_batch = tf.summary.merge_all(key='batch')

    display_message = ""

    sess.run(tf.local_variables_initializer())

    step = 0
    for i in range(epochs):
        train_result = {}
        total_img = 0
        print("Epoch: = {:d}".format(i))
        for image, label in get_batches_fn(batch_size):

            current_batch_size = len(label)
            total_img += current_batch_size

            if summary_batch is not None and summary_train is not None:
                _, loss, batch_b, train_b = sess.run([train_op, cross_entropy_loss, summary_batch, summary_train],
                                   feed_dict={input_image: image, correct_label: label, keep_prob: keep_prob_value, learning_rate: learning_rate_value})

                batch_sum = tf.Summary()
                batch_sum.ParseFromString(batch_b)
                train_sum = tf.Summary()
                train_sum.ParseFromString(train_b)
                for v in train_sum.value:
                    train_result[v.tag] = train_result.get(v.tag, 0) + v.simple_value * current_batch_size

                if writer is not None:
                    log_batch_history(sess, writer, batch_sum, step)
                    step += 1
            else:
                _, loss = sess.run([train_op, cross_entropy_loss],
                                            feed_dict={input_image: image, correct_label: label,
                                                       keep_prob: keep_prob_value, learning_rate: learning_rate_value})

            display_message = "Training loss: " + str(loss)

        if writer is not None:
            for key in train_result:
                train_result[key] /= total_img

            summary = tf.Summary()

            for (key, value) in train_result.items():
                summary.value.add(tag=key, simple_value=value)

            writer.add_summary(summary, i)

            display_message = display_message + " Training stat: " + str(train_result)

            if get_batches_fn_validate is not None:
                stat = log_validation_accuary(sess, i, writer, summary_validate, input_image, correct_label, keep_prob, get_batches_fn_validate,
                                  batch_size)
                display_message = display_message + " Validation stat: " + str(stat)

        # Save the training model
        if checkpoint_path is not None:
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_path)

            if i % 10 == 0:
                saver.save(sess, checkpoint_path + str(i)) # Save a copy also

        print(display_message)

tests.test_train_nn(train_nn)

from moviepy.editor import VideoFileClip
import scipy.misc
from PIL import Image

def make_real_video(sess, roi, img_shape, logits, keep_prob, input_image, input_path, output_path):
    def process_frame(img):
        # Input image is a Numpy array, resize it to match NN input dimensions
        street_img = Image.fromarray(img)
        img_orig_size = img.shape
        img = img[roi[0]:roi[2], roi[1]:roi[3], :]
        roi_size = (img.shape[0], img.shape[1])
        img_resized = scipy.misc.imresize(img, img_shape)
        #img_resized = img_resized / 127.5 -1

        # Process image with NN
        img_logits = sess.run([logits],
                               {keep_prob: 1.0, input_image: [img_resized]})
        img_softmax = helper.softmax(np.array(img_logits), axis=2)

        # Reshape to 2D image dimensions
        img_softmax = img_softmax[0][:, 1].reshape(img_shape[0],
                                                   img_shape[1])

        # Threshold softmax probability to a binary road judgement (>50%)
        segmentation = (img_softmax > 0.5).reshape(img_shape[0],
                                                   img_shape[1], 1)

        # Apply road judgement to original image as a mask with alpha = 50%
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask_resize_roi = scipy.misc.imresize(mask, roi_size)
        mask_resize = np.zeros((img_orig_size[0], img_orig_size[1], 4))
        mask_resize[roi[0]:roi[2], roi[1]:roi[3], :] = mask_resize_roi
        # draw a rectangle for ROI
        mask_resize[roi[0]:roi[2], roi[1], :] = [0, 0, 255, 127]
        mask_resize[roi[0]:roi[2], roi[3]-1, :] = [0, 0, 255, 127]
        mask_resize[roi[0], roi[1]:roi[3], :] = [0, 0, 255, 127]
        mask_resize[roi[2]-1, roi[1]:roi[3], :] = [0, 0, 255, 127]

        mask_resize = scipy.misc.toimage(mask_resize, mode="RGBA")

        #Resize the mask to the video original size and apply it on the video
        street_img.paste(mask_resize, box=None, mask=mask_resize)

        # Output image as a Numpy array
        img_out = np.array(street_img)
        return img_out

    # Process video frames
    video_outfile = output_path
    video = VideoFileClip(input_path)
    video_out = video.fl_image(process_frame)
    video_out.write_videofile(video_outfile, audio=False)

def save_model_to_pb(sess, export_dir):
    from tensorflow.python.saved_model import tag_constants

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], strip_default_attrs=True)

    builder.save()

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    log_dir = './log'
    train_model_dir = './trained_model'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 50
    batch_size = 32

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "train"

    print("Current mode: ",  mode)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    LOGDIR = os.path.join(log_dir, 'fcn8')
    model_path = os.path.join(train_model_dir, "model.ckpt")
    frozen_graph_path = os.path.join(train_model_dir, "graph_frozen.pb")
    optimize_graph_path = os.path.join(train_model_dir, "graph_optimized.pb")


    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        #get_batches_fn = helper.gen_batch_function_train(os.path.join(data_dir, 'data_road'), os.path.join(data_dir, 'cityscapes'), image_shape)
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road', 'training'), image_shape)
        get_batches_fn_val = helper.gen_batch_function_validate(os.path.join(data_dir, 'cityscapes'), image_shape)

        if mode == "train":
            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # Build NN using load_vgg, layers, and optimize function
            input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
            layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

            correct_label = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], num_classes),
                                           name='correct_label')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

            writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)

            # Train NN using the train_nn function
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, writer=writer, get_batches_fn_validate=get_batches_fn_val, checkpoint_path=model_path)

            # Save the model for future use
            saver = tf.train.Saver()
            saver.save(sess, model_path)
            #print("Model saved in path: %s" % save_path)

            # frozen the model
            utils.freeze_graph(train_model_dir, 'logits', frozen_graph_path)

            # optimized the model
            utils.optimize_graph(frozen_graph_path, optimize_graph_path, True, 'image_input', 'logits')

        elif mode == "test":
            dataset = sys.argv[2]

            gd = tf.GraphDef()
            g = sess.graph
            with tf.gfile.Open(optimize_graph_path, 'rb') as f:
                data = f.read()
                gd.ParseFromString(data)
            tf.import_graph_def(gd, name='')
            input_image = g.get_tensor_by_name('image_input:0')
            logits = g.get_tensor_by_name('logits:0')
            keep_prob = g.get_tensor_by_name('keep_prob:0')

            # Save inference data using helper.save_inference_samples
            if not dataset or dataset == 'kitti':
                helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
            elif dataset == 'cityscapes':
                helper.save_inference_samples_2(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
            else:
                print("Unknown dataset set, please use either kitti or cityscapes")

        elif mode == "video":
            if len(sys.argv) <8:
                print("main.py video <input video path> <output video path> <top> <left> <bottom> <right>")
            else:
                input = sys.argv[2]
                output = sys.argv[3]
                top = int(sys.argv[4])
                left = int(sys.argv[5])
                bottom = int(sys.argv[6])
                right = int(sys.argv[7])

                #saver = tf.train.Saver()
                #saver.restore(sess, tf.train.latest_checkpoint(train_model_dir))

                gd = tf.GraphDef()
                g = sess.graph
                with tf.gfile.Open(optimize_graph_path, 'rb') as f:
                    data = f.read()
                    gd.ParseFromString(data)
                tf.import_graph_def(gd, name='')
                input_image = g.get_tensor_by_name('image_input:0')
                logits = g.get_tensor_by_name('logits:0')
                keep_prob = g.get_tensor_by_name('keep_prob:0')

                make_real_video(sess, (top, left, bottom, right), image_shape, logits, keep_prob, input_image, input, output)
        else:
            print("Command unrecognized.")

if __name__ == '__main__':
    run()
