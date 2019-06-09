from sklearn.model_selection import train_test_split
import pdb
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
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

test_data = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTestRenameNominalValues.csv', header=None)


ctest = test_data.iloc[1:,41]
t = test_data.iloc[1:,0:41]

scaler = Normalizer().fit(t)
test_t = scaler.transform(t)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])



y_t = np.array(ctest)
y_test = y_t.astype(np.int)

# reshape input to be [samples, time steps, features]
x_train = np.reshape(test_t, (test_t.shape[0], 1, test_t.shape[1]))


batch_size = 32

# 1. define the network
model = Sequential()
model.add(SimpleRNN(4,input_dim=41))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# try using different optimizers and different optimizer configs
model.load_weights("kddresults/lstm1layer/checkpoint-53.hdf5")

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
y_train1 = y_test

y_pred = model.predict_classes(x_train)
# pdb.set_trace()
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("recall")
# pdb.set_trace()
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)
cm = metrics.confusion_matrix(y_train1, y_pred)
print("==============================================")
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
loss, accuracy = model.evaluate(x_train, y_train1)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

t_probs = model.predict_proba(x_train)
print(t_probs)
np.savetxt('probability.txt', t_probs)
print(t_probs.shape)