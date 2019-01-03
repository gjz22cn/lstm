import os
import sys
import numpy as np
import math
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

#skip_saved_model = False
skip_saved_model = True
g_modelDir = '../single_m'
g_stock = '002001.SZ'
g_path = './data'
g_modelFileName = g_modelDir + '/model_'+g_stock+'_save.h5'
g_preLen = 30


def prepare_dataset(data, time_steps):
    cnt = data.shape[0] - time_steps + 1
    data_x = data[:cnt]
    for i in range(1, time_steps):
        data_x = np.concatenate([data_x, data[i:i + cnt]], axis=1)

    data_x = data_x.reshape((cnt, time_steps, data.shape[1]))
    data_y = data[time_steps:, 3]

    return data_x[:-1], data_x[-1:], data_y


def prepare_model(time_steps, features):
    model = None
    if not skip_saved_model:
        if os.path.exists(g_modelFileName):
            model = load_model(g_modelFileName)

    if model is None:
        model = Sequential()
        model.add(LSTM(50, input_shape=(time_steps, features)))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

    return model


scaler = MinMaxScaler(feature_range=(0, 1))

file_name = '../data/'+g_stock+'.csv'
data = read_csv(file_name, header=0, usecols=['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount'],
                encoding='utf-8')
values = data.values.astype('float32')
g_count = values.shape[0] - g_preLen
dataset = scaler.fit_transform(values)
data_x, data_x_last, data_y = prepare_dataset(dataset, g_preLen)

train_size = int(data_x.shape[0] * 0.67)
train_x, test_x = data_x[:train_size], data_x[train_size:]
train_y, test_y = data_y[:train_size], data_y[train_size:]

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

model = prepare_model(g_preLen, train_x.shape[2])
model.fit(train_x, train_y, epochs=200, batch_size=64, validation_data=(test_x, test_y), verbose=1, shuffle=True)
model.save(g_modelFileName)

trainPredict = model.predict(train_x)
train_inverse_input = np.concatenate([dataset[:train_size, 0:3], trainPredict, dataset[:train_size, 4:]], axis=1)
trainPredict = scaler.inverse_transform(train_inverse_input)
trainPredict = trainPredict[:, 3]

testPredict = model.predict(test_x)
test_inverse_input = np.concatenate([dataset[train_size:g_count, 0:3], testPredict, dataset[train_size:g_count, 4:]],
                                    axis=1)
testPredict = scaler.inverse_transform(test_inverse_input)
testPredict = testPredict[:, 3]

lastPredict = model.predict(data_x_last)
last_inverse_input = np.concatenate([dataset[g_count:g_count + 1, 0:3], lastPredict, dataset[g_count:g_count + 1, 4:]],
                                    axis=1)
lastPredict = scaler.inverse_transform(last_inverse_input)
lastPredict = lastPredict[:, 3]
print("lastPredict=", lastPredict)

train_score = math.sqrt(mean_squared_error(train_y, trainPredict))
print("Train Score: %.2f RMSE" % train_score)
test_score = math.sqrt(mean_squared_error(test_y, testPredict))
print("Train Score: %.2f RMSE" % test_score)

plt.figure(figsize=(20, 6))
oriPlot = values[:, 3]
oriPlot = np.append(oriPlot, np.nan)

trainPredictPlot = np.empty_like(oriPlot)
trainPredictPlot[:] = np.nan
trainPredictPlot[g_preLen:g_preLen + train_size] = trainPredict

testPredictPlot = np.empty_like(oriPlot)
testPredictPlot[:] = np.nan
testPredictStartIdx = g_preLen + train_size
testPredictPlot[testPredictStartIdx:testPredictStartIdx + testPredict.shape[0]] = testPredict
#print("oriPlot:", oriPlot)
#print("trainPredictPlot:", trainPredictPlot)
#print("testPredictPlot:", testPredictPlot)

l1, = plt.plot(oriPlot, color='red', linewidth=3, linestyle='--')
l2, = plt.plot(trainPredictPlot, color='k', linewidth=2, linestyle='--')
l3, = plt.plot(testPredictPlot, color='g', linewidth=2, linestyle='--')
plt.ylabel('yuan')
plt.legend([l1, l2, l3], ('ori', 'train', 'test'), loc='best')
plt.title('Prediction')
plt.show()
