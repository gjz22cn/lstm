import os
import sys
import numpy as np
import datetime
import csv
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

skip_saved_model = False
# skip_saved_model = True
g_modelDir = '../single_m'
g_path = './data'
g_preLen = 30

output_file_name = '../predict/single_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
out = open(output_file_name, 'w', newline='')
csv_write = csv.writer(out, dialect='excel')

scaler = MinMaxScaler(feature_range=(0, 1))


def prepare_dataset(data, time_steps):
    cnt = data.shape[0] - time_steps + 1
    data_x = data[:cnt]
    for i in range(1, time_steps):
        data_x = np.concatenate([data_x, data[i:i + cnt]], axis=1)

    data_x = data_x.reshape((cnt, time_steps, data.shape[1]))
    data_y = data[time_steps:, 3]

    return data_x[:-1], data_x[-1:], data_y


def prepare_model(model_file_name, time_steps, features):
    model = None
    if not skip_saved_model:
        if os.path.exists(model_file_name):
            model = load_model(model_file_name)

    if model is None:
        model = Sequential()
        model.add(LSTM(50, input_shape=(time_steps, features)))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

    return model


def get_dataset_from_file(file_name):
    cols = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    data = read_csv(file_name, header=0, usecols=cols, encoding='utf-8')
    values = data.values.astype('float32')
    dataset = scaler.fit_transform(values)
    data_x, data_x_last, data_y = prepare_dataset(dataset, g_preLen)

    return data_x, data_x_last, data_y


def process_file(path, st_code):
    model_file_name = g_modelDir + '/model_' + st_code + '.h5'

    train_x, data_x_last, train_y = get_dataset_from_file(path)
    print(train_x.shape, train_y.shape)

    model = prepare_model(model_file_name, g_preLen, train_x.shape[2])
    model.fit(train_x, train_y, epochs=200, batch_size=64, verbose=0, shuffle=True)
    model.save(model_file_name)

    last_predict = model.predict(data_x_last)
    last_inverse_input = [[0, 0, 0, last_predict, 0, 0, 0]]
    last_predict = scaler.inverse_transform(last_inverse_input)
    predict = last_predict[:, 3][0]
    csv_write.writerow([st_code, predict])
    print("lastPredict=", predict)


def process_dir(dir):
    list = os.listdir(dir)
    count = 0
    skip_cnt = 0

    for i in range(skip_cnt, len(list)):
        if not list[i].endswith(".csv"):
            continue

        st_code = list[i][:-4]
        path = os.path.join(dir, list[i])
        if os.path.isfile(path):
            process_file(path, st_code)

        count += 1

        if count == 10:
            break



process_dir('../data')
out.close()
