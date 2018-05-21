# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

#### Implement

##### VGG architecture

VGG-16 is proposed to classify an image to a particular category. The following table
show VGG-16 internal layers.

Input: 224 x 224 x 3 (RGB)

Output: 1000 classes

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

##### FCN-8 architecture
Input: 576 x 160 x 3 (RGB)

Output: 576 x 160 x 1

Class: 0 = road, 1 = non-road

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
| | Soft max | 576 x 160 x 1 |

##### Data augmentation

* Not yet try any, will be done later

##### Training

* Freeze CNN layers weight, only train the last 2 convolution layers and fcn layers
* Train for 50 epoches
* Learning rate
* Drop rate

#### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

#### Result

Post a link (result) **

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.


