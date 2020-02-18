import csv, math
import matplotlib.pyplot as plt
import numpy as np
import process_gen
from sklearn.model_selection import train_test_split

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
         samples.append(line)     
  

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# Set our batch size
batch_size=32
assert np.asarray(samples).shape[0] == np.asarray(train_samples).shape[0]+ np.asarray(validation_samples).shape[0]

# compile and train the model using the generator function
train_generator = process_gen.generator(train_samples, batch_size=batch_size)
validation_generator = process_gen.generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3))) #160
# starts with five convolutional and maxpooling layers

model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(52, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator,steps_per_epoch=math.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps= math.ceil(len(validation_samples)/batch_size),epochs=5, verbose=1)

model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()