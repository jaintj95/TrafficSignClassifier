# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_plot.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./traffic_signs/11.jpg "Traffic Sign 1"
[image5]: ./traffic_signs/12.jpg "Traffic Sign 2"
[image6]: ./traffic_signs/13.jpg "Traffic Sign 3"
[image7]: ./traffic_signs/17.jpg "Traffic Sign 4"
[image8]: ./traffic_signs/18.jpg "Traffic Sign 5"
[image9]: ./traffic_signs/21.jpg "Traffic Sign 6"
[image10]: ./traffic_signs/31.jpg "Traffic Sign 7"
[image11]: ./traffic_signs/34.jpg "Traffic Sign 8"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jaintj95/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across various labels

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing the images, I first converted them to grayscale because training color images takes longer and is more prone to errors during evaluation. Also, from what I've read and understood from multiple videos and projects in the Self Driving and Deep Learning nanodegrees, color information isn't really important for training and classifying images. After converting images to grayscale, I normalized them as suggested in the lessons. From what I understand, normalizing the images apparantly helps the network learn faster and avoids the network from oscillating around (which usually happens when features are not scaled properly.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten				| Flatten 5x5x16 layer into 400 unit FC layer	|
| Fully connected		| 400 -> 120 hidden layer						|
| ReLU					|												|
| Dropout				| Dropout with 50% keep prob.					|
| Fully connected		| 120 -> 84 hidden layer						|
| ReLU					|												|
| Dropout				| Dropout with 50% keep prob.					|
| Fully connected o/p	| 84 -> 43 output layer 						|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the following parameters to train my model:
Optimizer: Adam
Learning Rate: 0.001
Batch size: 64
Epochs: 50
Dropout: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 99.2%
* Validation set accuracy of 97.0%
* Test set accuracy of 94.2% 

I initially chose to implement the basic LeNet architecture from the Nanodegree exercises.
The architecture worked fine but it couldn't achieve the desired target accuracy on test data.
I experimented with different learning rates, filter size, reducing and increasing the hidden layers, and implementing Dropout in the fully connected layers.
As per my experiments, dropout worked the best in significantly bumping up the model's accuracy, so I modified the initial architecture and added dropout to the Fully connected layers and used this architecture to train my model.
The reason dropout works so well is because it prevents the network from overfitting to training data by randomly dropping out neurons during training.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

The sixth image might be difficult to classify because it is under-represented in the training data and the model might have not learned it's features well during the training phase.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way  		| Right-of-way 									| 
| General Caution		| General Caution								|
| Priority Road			| Priority Road									|
| No Entry      		| No Entry  					 				|
| Double Curve     		| Children Crossing				 				|
| Yield         		| Yield     					 				|
| Turn left ahead  		| Turn left ahead				 				|
| Wild animals crossing	| Wild animals crossing			 				|


The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This compares favorably to the accuracy on the test set of 94.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the 1st image, the model is relatively sure that this is a Children Crossing sign (probability of 1.0), and the image contains a Double Curve sign. . The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Children Crossing								| 
| .00     				| Dangerous curve to the right					|
| .00					| Slippery road									|
| .00	      			| Beware of ice/snow			 				|
| .00				    | Bicycles crossing 							|


For the 2nd image, the model is relatively sure that this is a No Entry sign (probability of 1.0), and the image is indeed a No Entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry      								| 
| .00     				| Turn left ahead								|
| .00					| Stop      									|
| .00	      			| Keep right 					 				|
| .00				    | No passing        							|


For the 3rd image, the model is relatively sure that this is a Right-of-way sign (probability of 1.0), and the image is indeed a Right-of-way sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way  								| 
| .00     				| Beware of ice/snow							|
| .00					| Priority road									|
| .00	      			| Double curve 					 				|
| .00				    | Pedestrians       							|


For the 4th image, the model is relatively sure that this is a Priority Road sign (probability of 1.0), and the image is indeed a Priority Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority Road  								| 
| .00     				| Roundabout mandatory							|
| .00					| Ahead only									|
| .00	      			| Keep right 					 				|
| .00				    | Yield             							|


For the 5th image, the model is relatively sure that this is a Yield sign (probability of 1.0), and the image is indeed a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield         								| 
| .00     				| Speed limit (50km/h)							|
| .00					| Priority road									|
| .00	      			| Keep right 					 				|
| .00				    | Ahead only        							|


For the 6th image, the model is relatively sure that this is a Turn Left sign (probability of 1.0), and the image is indeed a Turn Left sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn Left     								| 
| .00     				| Keep right    								|
| .00					| No entry  									|
| .00	      			| Stop      					 				|
| .00				    | Yield             							|


For the 7th image, the model is relatively sure that this is a General Caution sign (probability of 1.0), and the image is indeed a Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General Caution								| 
| .00     				| Traffic signals 								|
| .00					| Pedestrians									|
| .00	      			| Right-of-way 					 				|
| .00				    | Roundabout mandatory 							|


For the 8th image, the model is relatively sure that this is a Wild Animals Crossing sign (probability of 1.0), and the image is indeed a Wild Animals Crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Wild Animals Crossing							| 
| .00     				| Double curve  								|
| .00					| Dangerous curve to the left					|
| .00	      			| Slippery road					 				|
| .00				    | Bicycles crossing  							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


