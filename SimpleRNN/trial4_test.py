from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pdb
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

testdata = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTestRenameNominalValues.csv', header=None)

# class column
C = testdata.iloc[1:,41]

# other columns
T = testdata.iloc[1:,0:41]


scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)


y_t = np.array(C)
y_test = y_t.astype(np.int)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 32

# 1. define the network
model = Sequential()
model.add(SimpleRNN(32,input_dim=41, return_sequences=True))  # try using a GRU
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=True))  # try using a GRU
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=True))  # try using a GRU
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=False))  # try using a GRU
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# try using different configs
model.load_weights("kddresults/lstm4layer/fullmodel/lstm4layer_model.hdf5")

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
y_train1 = y_test

y_pred = model.predict_classes(X_train)

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("recall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)
cm = metrics.confusion_matrix(y_train1, y_pred)
print("==============================================")
print(cm)
tp = cm[0][0]
fp = cm[0][1]
tn = cm[1][1]
fn = cm[1][0]
print("tp")
print(tp)
print("fp")
print(fp)
print("tn")
print(tn)
print("fn")
print(fn)

print("tpr")
tpr = float(tp)/(tp+fn)
print("fpr")
fpr = float(fp)/(fp+tn)
print("LSTM acc")
print(tpr)
print(fpr)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_train, y_train1)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))