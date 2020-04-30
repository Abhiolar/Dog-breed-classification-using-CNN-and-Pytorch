[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!


## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
    
__NOTE:__ if you are using the Udacity workspace, you *DO NOT* need to re-download the datasets in steps 2 and 3 - they can be found in the `/data` folder as noted within the workspace Jupyter notebook.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure you have already installed the necessary Python packages according to the README in the program repository.
5. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```


**PROJECT OVERVIEW**

This project focused on building a web app that can classify and identify different breeds of dogs using a convolutional neural network and comparing the model which we built from scratch and the pre-trained models on Torch vision. Deploying this model behind a web application will enable users to classify different dog breeds. A fun element of the app is that users can input human images and the model predicts the closest resemblance of dog breed to the human image.


**DATASETS AND INPUTS**

The first step in this stage is to download the dataset of the human and dog images. There are 13233 human images and 8351 dog images used in this project.

**ALGORITHM AND TECHNIQUES**

Convolutional Neural Network architecture is the main algorithm used for this project, the first step was to use pre-trained models such as in the haarcascade library to classify human and dog images. But a CNN is much prefered for the classification task of dog breeds as the model has to learn complex image patterns using high-frequency filters or kernels that aids in convolving an image onto a convolutional layer. The main goal was to build a CNN model from scratch after pre-processing of the inputs into image tensors and then passing these inputs through convolutional layers, max pooling layers to reduce the dimensions of the images before passing it through linear fully connected layers  and then output layer that predicts the probability of an image being of a certain class of dog breed.


**RESULTS**

11% test accuracy on test data for the built from scratch CNN architecture and 81% test accuracy on the resnet50 model using transfer learning technique.