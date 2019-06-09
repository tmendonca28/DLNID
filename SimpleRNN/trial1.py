import numpy as np
import pandas as pd
import pdb
from sklearn.preprocessing import Normalizer
from keras.utils import np_utils
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


train_data = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTrainRenameNominalValues.csv', header=None)
test_data = pd.read_csv('C:/Development/DLNID/Datasets/NSL-KDD/KDDTestRenameNominalValues.csv', header=None)

# iloc[rows to extract, columns to extract]
x = train_data.iloc[1:, 0:41]
# y shows duration for each record
y = train_data.iloc[1:, 41]

c = test_data.iloc[1:, 41]
t = test_data.iloc[1:, 0:41]


# Normalize the training data
scaler = Normalizer().fit(x)
train_x = scaler.transform(x)
# summarize the transformed data
np.set_printoptions(precision=3)
print(train_x[0:5, :])

# Normalize the test data
scaler = Normalizer().fit(t)
test_t = scaler.transform(t)
# summarize the transformed data
np.set_printoptions(precision=3)
print(test_t[0:5, :])

y_train = np.array(y)
y_test = np.array(c)

# reshape input to be [samples, time steps, features]
x_train = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
x_test = np.reshape(test_t, (test_t.shape[0], 1, test_t.shape[1]))

print(x_train.shape)

# batch_size: Integer or None. Number of samples per gradient update.If unspecified, batch_size will default to 32
# If dataset is small, it's best to make the batch_size equal to size of training data
batch_size = 32

# defining the network
model = Sequential()
model.add(SimpleRNN(4, input_dim=41))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.get_config())

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('csv_logger.csv',separator=',', append=False)
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1000, validation_data=(x_test, y_test),callbacks=[checkpointer,csv_logger])
model.save("kddresults/lstm1layer/fullmodel/lstm1layer_model.hdf5")

loss, accuracy = model.evaluate(x_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(x_test)
np.savetxt('kddresults/lstm1layer/lstm1predicted.txt', y_pred, fmt='%01d')