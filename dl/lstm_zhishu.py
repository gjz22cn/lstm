import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from pre_process import DataUtil
from dingding_msg import DingdingMsg
import matplotlib.pyplot as plt

class LstmZhiShu:
    def __init__(self):
        self.root_dir = '../'
        self.zhishus_dir = os.path.join(self.root_dir, 'zhishus')
        self.model_dir = os.path.join(self.root_dir, 'm_zhishu')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_class = 'm_zhishu'
        self.time_steps = 60
        self.features = 3
        self.predict_len = 1
        self.model = None
        self.zhishu_list = ['000001.SH', '000300.SH', '000905.SH', '399001.SZ', '399005.SZ', '399006.SZ', '399016.SZ',
                            '399300.SZ']

    def load_saved_model(self):
        print(self.model_file)
        if os.path.exists(self.model_file):
            self.model = load_model(self.model_file)
            return self.model

        return None

    def prepare_model(self, ts_code):
        st_code = ts_code.split('.')[0]
        model_file = os.path.join(self.model_dir, st_code + '_' + self.time_steps + '.h5')
        if os.path.exists(model_file):
            self.model = load_model(model_file)

        if self.model is None:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(self.time_steps, self.features), return_sequences=True))
            self.model.add(LSTM(100, return_sequences=False))
            self.model.add(Dense(self.predict_len))
            self.model.compile(loss='mae', optimizer='adam')

        return self.model

    def save_zhishu_model(self, ts_code):
        st_code = ts_code.split('.')[0]
        model_file = os.path.join(self.model_dir, st_code + '_' + self.time_steps + '.h5')
        self.model.save(model_file)

    def gen_dataset(self, df):
        data = df.values().astype('float32')
        data = self.scaler.fit_transform(data)

        cnt = data.shape[0] - self.time_steps - (self.predict_len-1)
        data_x = data[:cnt]
        for i in range(1, self.time_steps):
            data_x = np.concatenate([data_x, data[i:i + cnt]], axis=1)

        data_x = data_x.reshape((cnt, self.time_steps, data.shape[1]))
        data_y = data[self.time_steps:self.time_steps+cnt, 0:1]
        for i in range(1, self.predict_len):
            data_y = np.concatenate([data_y, data[self.time_steps+i:self.time_steps+i+cnt, 0:1]], axis=1)
        data_y = data_y.reshape((cnt, self.predict_len))

        return data_x, data_y

    def get_dataset(self, ts_code):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.zhishus_dir, ts_code + '.csv')
        cols = ['close', 'vol', 'amount']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')
        if df is None:
            return None, None

        if df.shape[0] < self.time_steps + self.predict_len:
            return None, None

        return self.gen_dataset(df)

    def shuffle_data_poll(self, data_x, data_y):
        return data_x, data_y
        '''
        len = data_x.shape[0]
        index = [i for i in range(len)]
        random.shuffle(index)
        return  data_x[index], data_y[index]
        '''

    def train_for_zhishu(self, ts_code):
        data_x, data_y = self.get_dataset(ts_code)
        if data_x is None or data_y is None:
            return

        data_s_x, data_s_y = self.shuffle_data_poll(data_x, data_y)
        train_size = int(data_x.shape[0] * 0.9)
        train_x, test_x = data_s_x[:train_size], data_s_x[train_size:]
        train_y, test_y = data_s_y[:train_size], data_s_y[train_size:]

        model = self.prepare_model(ts_code)
        model.fit(train_x, train_y, epochs=200, batch_size=64, validation_data=(test_x, test_y), verbose=1,
                  shuffle=True)

        self.save_zhishu_model(ts_code)

        train_p = model.predict(train_x)
        train_inverse_input = np.concatenate([train_p, train_x[:, 1:]], axis=1)
        train_p = self.scaler.inverse_transform(train_inverse_input)
        train_p = train_p[:, 0]

        test_p = model.predict(test_x)
        test_inverse_input = np.concatenate([test_p, test_x[:, 1:]], axis=1)
        test_p = self.scaler.inverse_transform(test_inverse_input)
        test_p = test_p[:, 0]
        self.show_plt(data_x[:, 0].flatten(), train_p.flatten(), test_p.flatten())

    def show_plt(self, ori, train, test):
        plt.figure(figsize=(20, 6))
        ori_plot = np.append(ori, np.nan)

        train_plot = np.empty_like(ori_plot)
        train_plot[:] = np.nan
        train_plot[self.predict_len:self.predict_len + train.shape[0]] = train

        test_plot = np.empty_like(ori_plot)
        test_plot[:] = np.nan
        test_idx_s = self.predict_len + train.shape[0]
        test_plot[test_idx_s:test_idx_s + test.shape[0]] = test

        l1, = plt.plot(ori_plot, color='red', linewidth=3, linestyle='--')
        l2, = plt.plot(train_plot, color='k', linewidth=2, linestyle='--')
        l3, = plt.plot(test_plot, color='g', linewidth=2, linestyle='--')
        plt.ylabel('yuan')
        plt.legend([l1, l2, l3], ('ori', 'train', 'test'), loc='best')
        plt.title('Prediction')
        plt.show()


if __name__ == '__main__':
    lstmZhiShu = LstmZhiShu()
    lstmZhiShu.train_for_zhishu('000001.SH')