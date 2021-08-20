#Get the dataset
! kaggle competitions download -c digit-recognizer

#Unzip them
! unzip /path/test.csv.zip
! unzip /path/train.csv.zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split



#Load CSV files
test = pd.read_csv('/content/test.csv')
train = pd.read_csv('/content/train.csv')
sample = pd.read_csv('/content/sample_submission.csv')

#Analysing Dataframes
test.head()
train.head()
sample.head()

#We can directly jump into spltting the data
#Defining train, test and labels
x_train = train.drop('label', axis=1)
y_train = train['label']
x_test = test

#Will check the shapes
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


#Converting them into an array
x_train = x_train.values.reshape(-1, 28, 28, 1)/255
x_test = x_test.values.reshape(-1, 28, 28, 1)/255

#One hot encoding the labels
y_train = to_categorical(y_train,10)

#Will see if everything is okay
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)



#Creating the model
model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()


#Will train
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=20, batch_size=50, verbose=2)

#Save the model
model.save('digit_recognizer.h5')

#Predicting using the given test dataset
results = model.predict(x_test)

results = np.argmax(results, axis=1)
results = pd.Series(results, name='label')




#Submission(Kaggle)
submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), results], axis=1)

submission.head()

#CSV to submit
submission.to_csv('submission.csv', index=False)
