import csv
import cv2
import numpy as np
lines =[]

import sklearn
def generator(samples, batch_size=32):
    """
	input the images in batch and flip the images
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

##reading one loop data 
with open('./sim_data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images_path = []
measurements = []
correction = 0.2
for line in lines:
  for i in range(3):
     path = line[i]
     images_path.append(path)
     
  center = float(line[3])
  left = center + correction
  right = center - correction
  measurements.append(center)
  measurements.append(left)
  measurements.append(right)

from sklearn.model_selection import train_test_split
samples = list(zip(images_path,measurements))
train_samples,valid_samples = train_test_split(samples,test_size=0.2)


train_generator = generator(train_samples,batch_size = 32)
valid_generator = generator(valid_samples,batch_size = 32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 

##model
model = Sequential()
model.add(Lambda(lambda x: x/225.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch =2*len(train_samples), validation_data = valid_generator, nb_val_samples = len(valid_samples)*2,nb_epoch=5, verbose=1)
model.save('model.h5')

##plot the train loss and validation loss
import matplotlib.pyplot as plt
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


