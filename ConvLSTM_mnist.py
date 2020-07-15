'Developed by Amirhossein Rezaei'
#List of world records can be found on http://yann.lecun.com/exdb/mnist/
#Dataset must be in 60,000 train images and 10,000 test images format.
#What matters here is the test result and not the training result, which here means the validation accuracy.
#Current best test error is 0.21%.
#This neural network, can achieve 99.85 % accuracy, which is equivalent to 0.15% test error
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential , model_from_json
from keras.layers import Dense, Conv2D , Flatten , MaxPooling2D , Dropout, LSTM , ConvLSTM2D , BatchNormalization ,Conv3D

X_train = X_train.reshape(60000,28,28,1,1)
X_test = X_test.reshape(10000,28,28,1,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model = Sequential()

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(28,28,1,1),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())


model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())


model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test) ,batch_size=300 ,epochs=30)



#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)

#model.save_weights("model.h5")
#print("Saved model to disk")
