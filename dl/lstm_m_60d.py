import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from pre_process import DataUtil

class LstmM60d:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_name = 'm_60d'
        self.root_dir = '../'
        self.model_dir = os.path.join(self.root_dir, self.model_name)
        self.model_file = os.path.join(self.model_dir, self.model_name+'.h5')
        self.stocks_dir = os.path.join(self.root_dir, 'stocks')
        self.time_steps = 60
        self.features = 6
        self.predict_len = 2
        self.skip_saved_model = False
        self.model = None
        self.data_util = DataUtil('../')
        self.stocks = []
        self.stocks_num = 0
        self.stock_idx = 0
        self.batch_size = 128
        self.train_pool_x = []
        self.train_pool_y = []
        self.data_idx = 0
        self.train_pool_size = 0
        self.steps_per_epoch = 0

    def prepare_model(self):
        if os.path.exists(self.model_file):
            self.model = load_model(self.model_file)

        if self.model is None:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(self.time_steps, self.features), return_sequences=True))
            self.model.add(LSTM(100, return_sequences=False))
            self.model.add(Dense(2))
            self.model.compile(loss='mae', optimizer='adam')

        return self.model

    def get_stocks(self):
        self.stocks = self.data_util.get_valid_single_stock_list()
        if self.stocks is None:
            return
        self.stocks_num = len(self.stocks)

    def gen_dataset(self, stock_data):
        data = self.scaler.fit_transform(stock_data)
        cnt = data.shape[0] - self.time_steps - 1
        data_x = data[:cnt]
        for i in range(1, self.time_steps):
            data_x = np.concatenate([data_x, data[i:i + cnt]], axis=1)

        data_x = data_x.reshape((cnt, self.time_steps, data.shape[1]))
        data_y = np.concatenate([data[self.time_steps:self.time_steps+cnt, 3:4],
                                 data[self.time_steps+1:self.time_steps+1+cnt, 3:4]], axis=1)
        data_y = data_y.reshape((cnt, self.predict_len))

        return data_x, data_y

    def load_next_stock(self):
        if self.stock_idx == self.stocks_num:
            return False

        st_code = self.stocks[self.stock_idx].split('.')[0]

        file = os.path.join(self.stocks_dir, st_code + '.csv')
        data = pd.read_csv(file, header=0, usecols=['open', 'high', 'low', 'close', 'vol', 'amount'], encoding='utf-8')
        if data is None:
            self.stock_idx += 1
            return False

        ava_data_cnt_delta = data.shape[0] - self.time_steps - 1
        if ava_data_cnt_delta < 1:
            self.stock_idx += 1
            return False

        values = data.values.astype('float32')
        data_x, data_y = self.gen_dataset(values)
        if self.train_pool_size == 0:
            self.train_pool_x = data_x
            self.train_pool_y = data_y
        else:
            self.train_pool_x = np.concatenate([self.train_pool_x[0-self.train_pool_size:], data_x], axis=0)
            self.train_pool_y = np.concatenate([self.train_pool_y[0-self.train_pool_size:], data_y], axis=0)

        self.train_pool_size += ava_data_cnt_delta
        self.data_idx = 0
        self.stock_idx += 1

    def generator(self):
        while True:
            while self.train_pool_size < self.batch_size:
                if self.stock_idx == self.stocks_num:
                    break
                self.load_next_stock()

            if self.train_pool_size >= self.batch_size:
                start = self.data_idx
                end = self.data_idx + self.batch_size
                self.data_idx += self.batch_size
                self.train_pool_size -= self.batch_size
                train_x = self.train_pool_x[start:end]
                train_y = self.train_pool_y[start:end]
                yield train_x, train_y

    def get_file_ava_num(self, file):
        count = len(["" for line in open(file, 'r', encoding='UTF-8')]) - 2
        ava = count - self.time_steps - 1
        if ava > 0:
            return ava

        return 0

    def get_steps_per_epoch(self):
        cnt = 0
        for stock in self.stocks:
            st_code = stock.split('.')[0]
            file = os.path.join(self.stocks_dir, st_code + '.csv')
            cnt += self.get_file_ava_num(file)

        return int(cnt/self.batch_size)

    def train(self):
        self.get_stocks()
        self.steps_per_epoch = self.get_steps_per_epoch()
        model = self.prepare_model()
        model.fit_generator(self.generator(),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=1,
                            verbose=1,
                            shuffle=True,
                            initial_epoch=0)

        model.save(self.model_file)


if __name__ == '__main__':
    lstmM60d = LstmM60d()
    lstmM60d.train()