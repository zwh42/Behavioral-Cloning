***Behavioral Cloning Project Summary*** 

---
**The goals / steps of this project are the following:**
* Use the [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road



[//]: # (Image References)

[model_summary]: ./images/model_summary.png "Model Summary"




---

**1. Submitted File List**

My project includes the following files:
* [model.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/model.py)  containing the script to create and train the model
* [drive.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/zwh42/Behavioral-Cloning/blob/master/model.h5) containing a trained convolution neural network 
* [ProjectSummary.md](https://github.com/zwh42/Behavioral-Cloning/blob/master/ProjectSummary.md) summarizing the results

**2. How to Run the Code with the Simulor**

Using the Udacity provided simulator (start the simulator and select autonomous mode) and my  [drive.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/drive.py) file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
(current model only work for track 1.)

**3. Model Architecture**

The overall strategy for deriving a model architecture was based on [nVidia's network](https://arxiv.org/abs/1604.07316), and made some changes by add a dropout layer to reduce the overfitting issue, since my train data may not as many as the original work.

The  [model.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

*Model Architecture and Training Strategy*
The model defination code is in [model.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/model.py) file's *model_setup()* function.
- the input images are firstly normalized and cropped in the model using a Keras lambda layer and cropping2d layer.
- The model consists of 4 convolution layers with 5x5/3x3 filter sizes and depths between 24 and 64, a max pooling layer is added after each convlution layer with pool size of 2x2.
- The model includes RELU activation layers to introduce nonlinearity. 
- To reduce overfitting in the model，a dropout layer is introduced with dropout ratio of 0.4.
- full connect layers contain 1164/100/50/10/1 neurons is added at the backend of the network to give the final ouput.  

The final model structure output by keras is shown as below:

![alt text][model_summary]

The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
The model used an adam optimizer, so the learning rate was not tuned manually.


**3. Training Data Collection, Processing and Augmentation**

- Training data was collected by running the simluator in **Trainning** mode and driving for serveral loops and record the driving. Some difficult locations(bridge, turing's) were collected with more rounds. More than 10, 000 training datas were collected.
- In the collected raw data, the appearance count of steering angle 0 is many times higher than other angles, which may create bias in the final model. To supress this issue, only a portion (no more than 4x of the 2nd most common steering angle) of angle = 0 input data was selected randomly. The code of this part is in the **data_preprocessing()** funtion of [model.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/model.py).   
- After processing, the effective trainning data count reduced by a large margin. To augment the traning data, the left and right camera images are also added, along with their steering angle, which has been modified by add some correction value (0.3). To further augment the data, all the images were flipped and the corresponding steering angle is also multipled by -1. The code of this part is in the **generator()** funtion of [model.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/model.py). 


**4. Creation of the Training Set & Training Process**

The augmented raw data was randomly splitted to 3 part. 10% for final test, then 20% of the remainning data was used for model validation. Then remains are used for model training.
The loss function of the training model is set to be "min squared error" and Adam optimizer is used. The mode is trainned for 10 epochs. The code of this part is in the **flow_setup()** funtion of [model.py](https://github.com/zwh42/Behavioral-Cloning/blob/master/model.py). 
