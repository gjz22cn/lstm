import os
import sys
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

skip_saved_model = False
g_path = './data'
g_modelFileName = '../comp_model/model_000001_save.h5'
g_preLen = 20

scaler = MinMaxScaler(feature_range=(0, 1))

def get_model(file_name):
    model = None
    if os.path.exists(file_name):
        model = load_model(file_name)

    return model

def get_dataset_from_file(file_name, time_steps):
    cols = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    data = read_csv(file_name, header=0, usecols=cols, encoding='utf-8')
    values = data.values.astype('float32')
    values = values[0-time_steps:]
    return scaler.fit_transform(values)

def prepare_input_dataset(data, time_steps):
    data_x = data[0:1]
    for i in range(1, time_steps):
        data_x = np.concatenate([data_x, data[i:i+1]], axis=1)

    data_x = data_x.reshape((1, time_steps, data.shape[1]))

    return data_x

def predict_by_file(file_name):
    dataset = get_dataset_from_file(file_name, g_preLen)
    data_x = prepare_input_dataset(dataset, g_preLen)
    print(data_x.shape)

    model = get_model(g_modelFileName)
    data_y = model.predict(data_x)
    print(data_y.shape, dataset[:, 0:3].shape)

    predict_inverse_input = np.concatenate([dataset[0:1, 0:3], data_y, dataset[0:1, 4:]], axis=1)
    predict = scaler.inverse_transform(predict_inverse_input)
    predict = predict[:, 3][0]
    print("predict=", predict)

file_name = '../data/000001.SZ.csv'
predict_by_file(file_name)







