# Semantic Segmentation

[//]: # (Image References)

[image1]: ./runs/1531164309.5460224/um_000004.png "kitti_sample"
[image2]: ./runs/1531164309.5460224/um_000060.png "kitti_sample"
[image3]: ./runs/1531164309.5460224/umm_000010.png "kitti_sample"
[image4]: ./runs/1531164309.5460224/umm_000059.png "kitti_sample"
[image5]: ./runs/1531164309.5460224/uu_000009.png "kitti_sample"
[image6]: ./runs/1531164309.5460224/uu_000095.png "kitti_sample"

[image7]: ./runs/1531165575.783992/berlin_000032_000019_leftImg8bit.png "cityscapes_sample"
[image8]: ./runs/1531165575.783992/bielefeld_000000_002308_leftImg8bit.png "cityscapes_sample"
[image9]: ./runs/1531165575.783992/bonn_000045_000019_leftImg8bit.png "cityscapes_sample"
[image10]: ./runs/1531165575.783992/leverkusen_000048_000019_leftImg8bit.png "cityscapes_sample"
[image11]: ./runs/1531165575.783992/mainz_000000_004237_leftImg8bit.png "cityscapes_sample"
[image12]: ./runs/1531165575.783992/munich_000026_000019_leftImg8bit.png "cityscapes_sample"

[image13]: ./writeup/training_loss.png "training loss"
[image14]: ./writeup/training_vs_validation_accuracy.png "training vs validation"

[image15]: ./writeup/cityscapes_setup.PNG "cityscapes dataset setup"

## Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

## Setup
### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

## Implement

### VGG architecture

VGG-16[1] is proposed to classify an image to a particular category.
The following table show VGG-16 internal layers.

* Input: 224 x 224 x 3 (RGB)
* Output: 1000 classes
* Image: [here](https://github.com/ymlai87416/CarND-Semantic-Segmentation/blob/master/images/vgg16_structure.png)

| VGG-16 Layer            | Output size  |
|:-----| :-----|
| Image input | 224 x 224 x 3 |
| Conv3-64 + RELU <br/> Conv3-64 + RELU | 224 x 224 x 64 <br/> 224 x 224 x 64 |
| Max pool (kernel=2, stride=2) | 112 x 112 x 64 |
| Conv3-128 + RELU <br/> Conv3-128 + RELU | 112 x 112 x 128 <br/> 112 x 112 x 128 |
| Max pool (kernel=2, stride=2, same) | 56 x 56 x 128 |
| Conv3-256 + RELU  <br/> Conv3-256 + RELU  <br/> Conv3-256 + RELU  | 56 x 56 x 256  <br/> 56 x 56 x 256  <br/> 56 x 56 x 256 |
| Max pool (kernel=2, stride=2, same) | 28 x 28 x 256 |
| Conv3-512 + RELU <br/> Conv3-512 + RELU <br/> Conv3-512 + RELU | 28 x 28 x 512 <br/> 28 x 28 x 512 <br/> 28 x 28 x 512 |
| Max pool (kernel=2, stride=2, same) | 14 x 14 x 512 |
| Conv3-512 + RELU <br/>Conv3-512 + RELU <br/>Conv3-512 + RELU | 14 x 14 x 512 <br/>14 x 14 x 512 <br/> 14 x 14 x 512 |
| Max pool (kernel=2, stride=2, same) | 7 x 7 x 512 |
| Flatten layer | 25088 |
| FC-4096 | 4096  |
| FC-4096 | 4096  |
| FC-1000 | 1000 |
| Soft max | 1000 |

### FCN-8 architecture

FCN[2] is proposed to solve semantic segmentation. It comes with several favor like FCN-8, FCN-16 and FCN-32.

In this project, FCN-8 is used, it takes the final layer and combines with predictions from 2 intermediate layers (pool3 and pool4),
and upsamples the combined result 8 times to obtain the result.

FCN-8 provides more accurate result than FCN-16 and FCN-32 by considering more output from intermediate layers.

* Input: 576 x 160 x 3 (RGB)
* Output: 576 x 160 x 1
* Class: 0 = road, 1 = non-road
* Image: [here](https://github.com/ymlai87416/CarND-Semantic-Segmentation/blob/master/images/fcn_structure.png)

Layers:

| Label | FCN-8 Layer | Output size  |
|:-----|:-----| :-----|
| | Image input | 576 x 160 x 3 |
| | Conv3-64 + RELU <br/> Conv3-64 + RELU | 576 x 160 x 64 <br/> 576 x 160 x 64 |
| | Max pool (kernel=2, stride=2) | 288 x 80 x 64 |
| | Conv3-128 + RELU <br/> Conv3-128 + RELU | 288 x 80 x 128 <br/> 288 x 80 x 128 |
| | Max pool (kernel=2, stride=2, same) | 144 x 40 x 128 |
| | Conv3-256 + RELU  <br/> Conv3-256 + RELU  <br/> Conv3-256 + RELU  | 144 x 40 x 256  <br/> 144 x 40 x 256  <br/> 144 x 40 x 256 |
| layer3_out | Max pool (kernel=2, stride=2, same) | 72 x 20 x 256 |
| | Conv3-512 + RELU <br/> Conv3-512 + RELU <br/> Conv3-512 + RELU | 72 x 20 x 512 <br/> 72 x 20 x 512 <br/> 72 x 20 x 512 |
| layer4_out | Max pool (kernel=2, stride=2, same) | 36 x 10 x 512 |
| | Conv3-512 + RELU <br/>Conv3-512 + RELU <br/>Conv3-512 + RELU | 36 x 10 x 512 <br/>36 x 10 x 512 <br/> 36 x 10 x 512 |
| | Max pool (kernel=2, stride=2, same) | 18 x 5 x 512 |
| layer7_out | Conv7-4096 + RELU + dropout <br/> Conv1-4096 + RELU + dropout | 18 x 5 x 4096 <br/> 18 x 5 x 4096 |
| Intermediate_1 <- layer4_out | Conv1-2, stride=2 | 36 x 10 x 2  |
| Intermediate_1 <- layer7_out | Conv1-2, stride=2 <br/> Deconv4-2, stride=2 | 18 x 5 x 2 <br/>36 x 10 x 2  |
| Intermediate_2 <- layer3_out| Conv1-2, stride=2 | 72 x 20 x 2 |
| Intermediate_2 <- Intermediate_1| Deconv1-2, stride=2 | 72 x 20 x 2 |
| fcn_output <- Intermediate_2| Deconv16-2, stride = 8 | 576 x 160 x 2 |
| | Softmax | 576 x 160 x 1 |

### Training

#### Data

Both Kitti road dataset (289 images) and cityscapes dataset (2975 images) is used to train the neural network.
the validation set of cityscapes (500 images) is used as the validation set.

The training image is augmented before it is used to train the neural network.
The implementation is the function `augment()` under `helper.py`

The following operations are applied to the image during augmentation.

| Operation | Function call | Chance  |
|:-----|:-----| :-----|
| Change the brightness of a image| `apply_brightness_augmentation()` in `helper.py` | 100% |
| Apply shadow| `apply_random_shadow()` in `helper.py` | 25% |
| Translation the image along x-y axis| `apply_translation()` in `helper.py` | 50% |
| Flip the image horizontally| `cv2.flip(img_trans, 1)` in `helper.py`  | 25% |

##### To setup the project to use cityscapes dataset

Current project setup only trains a neural network on Kitti dataset only for grading purpose.

If you want to use also the cityscapes dataset for training, please do the following

* Put the cityscapes data under `./data` like the following:

![alt text][image15]

Download both leftImg8bit and gtFine dataset from [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
and extract the content to `./data` folder.

* Change the get_batches_fn to use gen_batch_function_train() instead of gen_batch_function()
```
get_batches_fn = helper.gen_batch_function_train(os.path.join(data_dir, 'data_road'), os.path.join(data_dir, 'cityscapes'), image_shape)
#get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road', 'training'), image_shape)
````

#### Layers

Only FCN layers and the last 2 convolution layers (fc6 and fc7) is trained. Other layers are frozen, and act as the
feature extractor.

#### Epoch, learning rate ,and dropout

This training uses Adam Optimizer with learning rate=`0.0001`,
dropout rate=`0.5` for dropout layer after fc6 and fc7, and train the neural network for `50` epochs.

The training is done on Nvidia GeForce GTX 1080Ti, the time for each epoch is around 7 minutes 10 seconds. (using both Kitti and cityscapes)

| Loss | Accuracy |
|:-----|:-----|
|![alt text][image13]|![alt text][image14]|

From the accuracy graph, the training accuracy increase with the validation accuracy, which means the network is not overfitting the training
set at 50th epoch.

### Inference rate

Using the trained model directly result in inference rate of 1 frame / second, which is too slow the for any practical use.
To improve the inference speed, the model is further enhanced.

In `utils.py`, there are 2 functions for this purpose.
`freeze_graph()` is a function to freeze the graph, while `optimize_graph()` a function to optimize the graph for inference.

Here is the performance comparison for applying freeze graph and optimized graph operation:

| Version | Inference speed |
|:-----|:-----|
|original version|~1 frame/s|
|frozen graph|~5 frames/s|
|optimized graph|~5 frames/s|

As the image is processed by CPU before it is sent to GPU for inference, CPU may become the bottleneck of inference rate.
The throughput could be improved if both CPU and GPU process run in parallel.

## Run

As neural network training is time-consuming, it is better for `main.py` script to work on a task once at a time.

For training the neural network, run the following command

```
python main.py train OR python main.py
```

For generate labeled images of test set, run the following command
```
python main.py test kitti OR python main.py test cityscapes
```
The resulting images are stored under `./run`

For generate labeled video, run the following command
```
python main.py video <input video path> <output video path> <top> <left> <bottom> <right>
```

The script accepts a file-path of an input video and the output file-path, and also a rectangle representing the Region of interest (ROI).
It is best to have the dimension of ROI nears to the multiple of 576 x 160, ROI is drawn as a blue rectangle in the output video.

Here is an example.
```
video ./video/challenge_video.mp4 ./video/challenge_video_output.mp4 314 0 670 1280
```

## Result

### Kitti road set result

The trained neural network labels road pixels in the test image set of the Kitti road dataset.

The resulting images are at `runs/1531164309.5460224`

| Sample images           | Sample images |
|:-------------:| :-----:|
| ![alt text][image1] | ![alt text][image2] |
| ![alt text][image3] | ![alt text][image4] |
| ![alt text][image5] | ![alt text][image6] |


### Cityscapes dataset result

Road pixels in the test image set of the cityscapes dataset are labeled and the resulting images are at `runs/1531165575.783992`


| Sample images           | Sample images  |
|:-------------:| :-----:|
| ![alt text][image7] | ![alt text][image8] |
| ![alt text][image9] | ![alt text][image10] |
| ![alt text][image11] | ![alt text][image12] |

### Video result

I have also used the trained neural network to label road pixel of the following videos:

| Input video           | Output video  |
|:-------------| :-----|
| ./video/project_video.mp4 | ./video/project_video_output.mp4 [Youtube](https://youtu.be/pY_yx5fJctA) |
| ./video/challenge_video.mp4 | ./video/challenge_video_output.mp4 [Youtube](https://youtu.be/kUz1RNT5TAE)|
| ./video/harder_challenge_video.mp4 | ./video/harder_challenge_video_output.mp4 [Youtube](https://youtu.be/kbh5LZV2Obs)|


## Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)


## Reference
[1] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

[2] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

