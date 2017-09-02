# Behavioral_Cloning
Training a deep neural network to drive a simulated car. 

###Model Architecture and Training Strategy

### Model Arcitecture 
I Chose to use NVIDIA's model for autonomous cars, because if it workes for them then it is probably a good starting point. It consists of 4 convolutional layers with Rectified Linear Activation fucntions and 3 fully connected layers followed by the output layer. 
### Attempts to reduce over fitting
The model contains dropout layers in order to reduce overfitting (model.py lines 104,106).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 50,51 or line 119 ). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also recorded laps going counter-clockwise.  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

To get the project up and running I initally tried the Lenet architecture that I used for sign classification, but this approach didn't seem to work well, so I studied the NVIDIA model from the lectures and settled upon using it. The only modification I made was introducing dropout between the first two fully connected layers. This helped with over fitting and seemed to smooth out some of the jerky steering learned from the provided data. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added more training data to the training data set that addressed similar situations to the one where the car went off the track. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
|Lambda Layer| Normalize picture -1,1 
|Cropping layer | Crop out unneccesary parts of photo 
||
| Input         		| 160x320x3 RGB image 
| Convolution    5x5  	| RELU activation 	|
|  Convolution 5x5	    | RELU activation 
| Convolution 5x5	    | RELU activation     			|
| Convolution 3x3	    | RELU activation     			|
|Flatten|
| Fully connected		|  Output 100    |	
| Droput | keep probability 0.5 |
| Fully connected		|  Output 50    |
| Droput | keep probability 0.5 |
| Fully connected		|  Output 10    |
||
| Output Layer | 

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track two using center lane driving.  One clockwise, and the other counterclockwise. The third lap focused on very slowly and smmotly going around corners. 

I then recorded data of the car near the edges of the road or with obstacles in front with the steering wheel sharply turned towards the center of the road. 

To augment the data set, I  flipped images and angles so that the model wouldn't develop a bias for tuning one direction over the other which should help it generalize about curves, and corners. 

I also converted the images to RGB because OpenCV is BGR by default. 

After the collection process, I had 89000 number of data points. I then preprocessed this data by normalizing the images to pixel values between -1,1 in keras lambda layer, and set the top 70, and bottom 20 pixels of each image to black in a Karas cropping layer.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4-5 as evidenced by minimum difference between validation and training loss.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

I added a generator into the project but never used it as it was much slower than running without. It should function the same. 
