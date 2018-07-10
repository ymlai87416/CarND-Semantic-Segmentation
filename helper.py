import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def get_label_kitti(gt_image, background_color):
    gt_road = np.all(gt_image == background_color, axis=2)
    gt_road = gt_road.reshape(*gt_road.shape, 1)
    gt_road = gt_road.astype(np.uint8)
    gt_road = 1-gt_road
    return gt_road

def get_label_cityscapes(gt_image, road_label):
    gt_road = (gt_image == road_label)
    gt_road = gt_road.reshape(*gt_road.shape, 1)
    gt_road = gt_road.astype(np.uint8)
    return gt_road

cityscpaes_roi = (216, 0, 834, 2048)

def corp_cityscapes_roi(image, roi):
    if len(image.shape) == 3:
        return image[cityscpaes_roi[0]:cityscpaes_roi[2], cityscpaes_roi[1]:cityscpaes_roi[3], :]
    else:
        return image[cityscpaes_roi[0]:cityscpaes_roi[2], cityscpaes_roi[1]:cityscpaes_roi[3]]

def apply_random_shadow(image):
    #
    # Add a random shadow to a BGR image to pretend
    # we've got clouds or other interference on the road.
    #
    rows, cols, _ = image.shape
    top_y = cols * np.random.uniform()
    top_x = 0
    bot_x = rows
    bot_y = cols * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = (shadow_mask == 1)
        cond0 = (shadow_mask == 0)
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def apply_brightness_augmentation(image):
    #
    # expects input image as BGR, adjusts brightness to
    # pretend we're in different lighting conditions.
    #
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image2 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image2


def apply_translation(image, label, translation_range):
    #
    # Shift image up or down a bit within trans_range pixels,
    # filling missing area with black.  IMG is in BGR format.
    #
    rows, cols, _ = image.shape
    tr_x = translation_range * np.random.uniform() - translation_range / 2
    tr_y = 10 * np.random.uniform() - 10 / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    img_tr = cv2.warpAffine(image, trans_m, (cols, rows))
    label_tr = cv2.warpAffine(label, trans_m, (cols, rows))
    return img_tr, label_tr


def augment(image_raw, label):
    img = apply_brightness_augmentation(image_raw)

    if np.random.randint(4) == 0:
        img_shadows = apply_random_shadow(img)
    else:
        img_shadows = img

    if np.random.randint(2) == 0:
        img_trans, label_trans = apply_translation(img_shadows, label, 25)
    else:
        img_trans, label_trans = img_shadows, label

    if np.random.randint(4) == 0:
        img_flip = cv2.flip(img_trans, 1)
        label_flip = cv2.flip(label_trans, 1)
    else:
        img_flip, label_flip = img_trans, label_trans

    return img_flip, label_flip

def gen_batch_function_validate(cityscapes_data_folder, image_shape):
    """
        Generate function to create batches of training data
        :param data_folder: Path to folder that contains all the datasets
        :param image_shape: Tuple - Shape of image
        :return:
        """

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths_cityscapes = glob(
            os.path.join(cityscapes_data_folder, 'leftImg8bit', 'val', '**', '*_leftImg8bit.png'))
        label_paths_cityscapes = {
            re.sub(r'_gtFine_labelIds.png', '_leftImg8bit.png', os.path.basename(path)): path
            for path in glob(os.path.join(cityscapes_data_folder, 'gtFine', 'val', '**', '*_gtFine_labelIds.png'))}

        road_label_cityscaper = 7

        image_paths = image_paths_cityscapes

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths_cityscapes.get(os.path.basename(image_file), None)

                image = scipy.misc.imresize(corp_cityscapes_roi(scipy.misc.imread(image_file), cityscpaes_roi),
                                            image_shape)
                gt_image = scipy.misc.imresize(
                    corp_cityscapes_roi(scipy.misc.imread(gt_image_file), cityscpaes_roi), image_shape)
                gt_road = get_label_cityscapes(gt_image, road_label_cityscaper)

                gt_image = np.concatenate(((gt_road==0).astype(np.uint8), gt_road), axis=2)

                #image = image / 127.5 - 1.
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn

def gen_batch_function_train(kitti_data_folder, cityscapes_data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths_kitti = glob(os.path.join(kitti_data_folder, 'training', 'image_2', '*.png'))
        label_paths_kitti = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(kitti_data_folder, 'training', 'gt_image_2', '*_road_*.png'))}

        image_paths_cityscapes = glob(
            os.path.join(cityscapes_data_folder, 'leftImg8bit', 'train', '**', '*_leftImg8bit.png'))
        label_paths_cityscapes = {
            re.sub(r'_gtFine_labelIds.png', '_leftImg8bit.png', os.path.basename(path)): path
            for path in glob(os.path.join(cityscapes_data_folder, 'gtFine', 'train', '**', '*_gtFine_labelIds.png'))}

        # print('Debug ' + os.path.join(cityscapes_data_folder, 'leftImg8bit', 'train', '**', '_leftImg8bit.png'))
        # print('Debug ' + str(len(image_paths_cityscapes)))

        background_color_kitti = np.array([255, 0, 0])
        road_label_cityscaper = 7

        image_paths = image_paths_kitti + image_paths_cityscapes

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                # print('Debug: ' + image_file)

                gt_image_file = label_paths_cityscapes.get(os.path.basename(image_file), None)

                if gt_image_file is not None:
                    image = scipy.misc.imresize(corp_cityscapes_roi(scipy.misc.imread(image_file), cityscpaes_roi),
                                                image_shape)
                    gt_image = scipy.misc.imresize(
                        corp_cityscapes_roi(scipy.misc.imread(gt_image_file), cityscpaes_roi), image_shape)
                    gt_road = get_label_cityscapes(gt_image, road_label_cityscaper)
                else:
                    gt_image_file = label_paths_kitti[os.path.basename(image_file)]
                    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                    gt_road = get_label_kitti(gt_image, background_color_kitti)

                image_a, gt_road_a = augment(image, gt_road)
                gt_road_a = np.atleast_3d(gt_road_a)

                gt_image = np.concatenate(((gt_road_a==0).astype(np.uint8), gt_road_a), axis=2)
                #image_a = image_a / 127.5 - 1.

                images.append(image_a)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        #image = image / 127.5 - 1.

        im_logits = sess.run(
            [logits],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = softmax(np.array(im_logits), axis=2)

        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def gen_test_output2(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder,'leftImg8bit', 'test', '**', '*_leftImg8bit.png')):
        image = scipy.misc.imresize(corp_cityscapes_roi(scipy.misc.imread(image_file), cityscpaes_roi), image_shape)
        #image = image / 127.5 - 1.

        im_logits = sess.run(
            [logits],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = softmax(np.array(im_logits), axis=2)

        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples_2(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output2(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'cityscapes'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)