import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# Set to true if you wish to use generator rather than load everything into memory
GENERATOR = False
# steering correction, works best with negative value, I believe due to training data going counter-clockwise
correction = -0.3


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples) # Shuffle for regularisation.
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                filenameC = '../data/IMG/' + batch_sample [0].split('/')[-1]

                imageC = cv2.imread(filenameC)
                imageC = cv2.cvtColor(imageC, cv2.COLOR_BGR2RGB)

                filenameL = '../data/IMG/' + batch_sample [1].split('/')[-1]

                imageL = cv2.imread(filenameL)
                imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)

                filenameR = '../data/IMG/' + batch_sample [2].split('/')[-1]

                imageR = cv2.imread(filenameR)
                imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)

                images.extend([imageC, imageL, imageR])
                images.extend([cv2.flip(imageC, 1), cv2.flip(imageL, 1), cv2.flip(imageR, 1)])

                measurement = float(line[3])
                steering_left = measurement + correction
                steering_right = measurement - correction

                measurements.extend([measurement, steering_left, steering_right])
                measurements.extend([-measurement, -steering_left, -steering_right])
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train,y_train)

lines = []

with open("../data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)  # Remove first element from Udacity data, causes crash.

if GENERATOR:
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = generator(train_samples)          # set up generator callbacks for loading data at runtime.
    validation_generator = generator(validation_samples) # to be fed into Keras model.fit()
else:
    images = [] # Image samples
    measurements = [] # steering values at each image.

    for line in lines:

        if 1:  # -6 < float(line[3]) < 6:
            filenameC = '../data/IMG/'+line[0].split('/')[-1]

            imageC = cv2.imread(filenameC)
            imageC = cv2.cvtColor(imageC, cv2.COLOR_BGR2RGB)

            filenameL = '../data/IMG/' + line[1].split('/')[-1]

            imageL = cv2.imread(filenameL)
            imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)

            filenameR = '../data/IMG/' + line[2].split('/')[-1]

            imageR = cv2.imread(filenameR)
            imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)

            images.extend([imageC,imageL,imageR])
            images.extend([cv2.flip(imageC, 1),cv2.flip(imageL, 1),cv2.flip(imageR, 1)])

            measurement = float(line[3])
            steering_left = measurement + correction
            steering_right = measurement - correction

            measurements.extend([measurement,steering_left,steering_right])
            measurements.extend([-measurement,-steering_left,-steering_right])

    X_train = np.array(images)
    y_train = np.array(measurements)


model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3))) # normalization
model.add(Cropping2D(cropping=((70, 20), (0, 0)))) # cropping to remove unnecessary data.

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))


model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5)) # dropout to help with over fitting
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))

model.add(Dense(1))
# no need for activation because the desired data will be a label for steering, not a probability or classification.

model.compile(loss='mse', optimizer='adam') # learning rate tuned inside optimizer.
if GENERATOR:
    history_object = model.fit_generator(train_generator,
                                         samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples=len(validation_samples),
                                         nb_epoch=5)
else:
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)


model.save('../CarND-Behavioral-Cloning-P3/model.h5')


# Plot for visual of training vs validation.
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

