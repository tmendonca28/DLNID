from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

traindata = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTrainRenameNominalValues.csv', header=None)
testdata = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTestRenameNominalValues.csv', header=None)


X = traindata.iloc[1:,0:41]
Y = traindata.iloc[1:,41]
C = testdata.iloc[1:,41]
T = testdata.iloc[1:,0:41]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 32

# 1. define the network
model = Sequential()
model.add(SimpleRNN(32,input_dim=41, return_sequences=True))  # try using a GRU
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=True))
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=True))
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm4layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True,monitor='val_acc',mode='max')
csv_logger = CSVLogger('csv_logger4.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
model.save("kddresults/lstm4layer/fullmodel/lstm4layer_model.hdf5")

loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(X_test)
np.savetxt('kddresults/lstm4layer/lstm4predicted.txt', y_pred, fmt='%01d')