# **Behavioral Cloning** 

## Writeup Template

###  This writeup presents the methods, challenges and process used to complete the Behavioral Cloning project by implementing keras based - convolutional neural networks for prediction. In the project, I trained a Nvidia based neural network architecture to mimic the driving properties of a simulated car, validated the network model and tested it in the provided simulated environment. The test showed that the car was able to drive itself for over 2 laps of testing on track 1.
---
**Behavioral Cloning -Files** The main project is presented in the model.py file and the compiled network model is in the model.h file. Due to the size of this file (107Mb), it is compressed in a zip folder for submission as model.zip. Drive.py also included, is modified to include the image processing carried out on images in the training phase and the video.mp4 shows the video of the car driving itself. 

**Process to build a Traffic Sign Recognition System** To successfully complete the project, I followed the objectives given as a guide. I also made use of provided software snippets and slightly modified the Nvidia neural network architecture for self driving cars.
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
For some reasons, the model.h5 file is very large so I had to compress it before submission. Therefore, this file needs to be decompressed before used. However, a copy of the submission set is included in the workspace and the version of model.h5 there is not compressed.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. This file works with the process_gen file. In the process_gen.py file, I wrote a pipeline that used generators to read training data in batches and carryout image processing on them. The generator function was written using the example presented in the lecture and modified to suit the image processing task. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. To get the validation data, I split the total data in a ratio 8:2, 80% of the data for training while 20% for validating. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I used a modified version of the Nvidia self-driving car model. However, as part of my learning, I used 6 instead of 5 convolutional network layers with maxpooling. In order to make this work, I had to increase the size of the images after cropping so the iteration do not abruptly stop with a negative data error. In addition to this I used five fully connected layers as discribed in the Nvidia self driving car architecture (model.py lines 30 - 53) 

The model includes RELU layers to introduce nonlinearity (code line 35), and the data is normalized in the model using a Keras lambda layer (code line 31). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 42, 57, 61). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15, 21 and 22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I used the data provided as I could not get my data to complete a lap. I did my research on this and fellow students experience this and say the data should be recorded using a joystick and not a keyboard. I could not get my new version of xbox wireless controllers to work with the software. I had to use the sample data which worked well.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia self driving car architecture. I thought this model might be appropriate because it was recommended for this purpose.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by including dropouts at various points. I also increased the amount of validation data from 10% to 20%

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 30 - 53) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 80, 160, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 80, 160, 24)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 79, 159, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 40, 80, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 40, 80, 36)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 39, 79, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 20, 40, 48)        43248     
_________________________________________________________________
activation_3 (Activation)    (None, 20, 40, 48)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 40, 48)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 19, 39, 48)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 20, 52)        62452     
_________________________________________________________________
activation_4 (Activation)    (None, 10, 20, 52)        0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 9, 19, 52)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 9, 19, 64)         30016     
_________________________________________________________________
activation_5 (Activation)    (None, 9, 19, 64)         0         
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 8, 18, 64)         0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 18, 64)         36928     
_________________________________________________________________
activation_6 (Activation)    (None, 8, 18, 64)         0         
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 7, 17, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 7616)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              8866188   
_________________________________________________________________
activation_7 (Activation)    (None, 1164)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1164)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
activation_8 (Activation)    (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
activation_9 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_10 (Activation)   (None, 10)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
_________________________________________________________________

Total params: 9,184,363
Trainable params: 9,184,363
Non-trainable params: 0

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the provided sample data for training. I read-in the lines from the csv file, randomly shuffled them, then read in the center, left and right images, and the steering angle of the center image. I added a steering angle correction of 0.3 to the left camera images and subtracted a steering angle correction of 0.3 from the right camera images. I placed eachof these in arrays and processed them. The image processing pipeline was developed by reading various writeups of former students ( https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk). 


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by increasing this increases the data loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Challenges
This has so far been the most challenging project yet. Given the GPU constraints, the long training time and testing the car on the simulation, the wait to watch it work for a while only to fail later. The major challenges with the project was in implementing a generator -which is quite new to me, processing the images to ease learning, reading in file -the format changes when I use the sample training data or when I use my own generated data. Also, the GPU used some deprecated functions and this had to be updated.
