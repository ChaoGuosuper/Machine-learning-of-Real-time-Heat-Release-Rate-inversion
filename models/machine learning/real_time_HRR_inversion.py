from math import sqrt
from numpy import concatenate
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
import time


# load dataset
dataset = read_csv('test_datasets_3.csv', header=0, index_col=0)
# dataset = read_csv('test_datasets_2.csv', header=0, index_col=0)
# dataset = read_csv('test_datasets_3.csv', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')
test = values[0:1001, :]

# data  normalization
scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test)
testinput = test[:, :-1]
testlabel = test[:, -1]
n_hours = 1
n_features = 13
test_X, test_y = testinput[:, :], testlabel[:]
model = load_model('lstmmodel5.h5')
# model=joblib.load('svr.pkl')

# The additional setup is required for lstm network
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))


# prediction
t0 = time.time()
yhat = model.predict(test_X)
yhat = yhat.reshape((len(yhat), 1))
time_consuming = time.time() - t0
print('time_consuming:',time_consuming)

# inverse_transform
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
inv_yhat = concatenate((test_X[:, :],yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, :],test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]


# RMSE MSE R2
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mse = mean_squared_error(inv_y, inv_yhat)
R2 = r2_score(inv_y, inv_yhat)

print('Test RMSE: %.3f' % rmse)
print('Test MSE: %.3f' % mse)
print('Test R2: %.3f' % R2)

plt.plot(inv_y,"b--",label="real")
plt.plot(inv_yhat,"r--",label="prediction")
plt.legend()
plt.show()