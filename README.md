# Behavioral_Cloning     
![gif](https://github.com/Yunying-Chen/Behavioral_Cloning/blob/master/image/auto.gif)    
1.Use the simulator to collect data of good driving behavior      
2.Build, a convolution neural network in Keras that predicts steering angles from images      
3.Train and validate the model with a training and validation set      
4.Test that the model successfully drives around track one without leaving the road


## Environment
-Python     
-Opencv      
-Sklearn      
-Keras      


## Files
-model.py containing the script to create and train the model       
-drive.py for driving the car in autonomous mode      
-model.h5 containing a trained convolution neural network      




the car can be driven autonomously around the track by executing         
\'python drive.py model.h5\'

## Network
My network is based on Nvidia net and its architecture is as follows:                
![network](https://github.com/Yunying-Chen/Behavioral_Cloning/blob/master/image/Network.jpg)

## Performance
The data was randomly shuffled the data set and 20% of the data was put into a validation set.       
The number of epochs was 5 and the loss changed as follows:       
![loss](https://github.com/Yunying-Chen/Behavioral_Cloning/blob/master/image/loss.jpg)

