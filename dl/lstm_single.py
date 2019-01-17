import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from DownloadClient import DownloadClient
import datetime
from pre_process import DataUtil

skip_saved_model = False
# skip_saved_model = True
g_modelDir = '../single_m'
g_path = './data'
g_preLen = 10


class SingleLstm:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.root_dir = '../'
        self.single_dir = os.path.join(self.root_dir, 'single')
        self.single_data_dir = os.path.join(self.root_dir, 'single_d')
        self.data_dir = os.path.join(self.root_dir, 'single_d_10')
        self.model_file = os.path.join(self.single_dir, 'single_m.h5')
        self.time_steps = 10
        self.features = 7
        self.skip_saved_model = False
        self.model = None
        self.data_util = DataUtil('../')

    def get_valid_single_stock_list(self):
        file_path = os.path.join(self.single_data_dir, 'single_stock_list.csv')
        df = pd.read_csv(file_path, header=0, usecols=['ts_code'], encoding='utf-8')
        return df.values.flatten()

    def gen_dataset(self):
        stocks = self.get_valid_single_stock_list()
        dataset = None
        for stock in stocks:
            st_code = stock.split('.')[0]
            file = os.path.join(self.data_dir, st_code + ".csv")
            df = pd.read_csv(file, header=0, encoding='utf-8')
            if df is None:
                continue

            values = df.values[:, 1:].astype('float32')
            dataset_one = self.scaler.fit_transform(values)

            if dataset is None:
                dataset = dataset_one
            else:
                dataset = np.concatenate([dataset, dataset_one], axis=0)

        print(dataset.shape)
        return dataset[:, :-1].reshape((-1, self.time_steps, self.features)), dataset[:, -1:]

    def load_p230_model(self):
        if os.path.exists(self.model_file):
            self.model = load_model(self.model_file)
            return self.model

        return None

    def prepare_model(self):
        if not self.skip_saved_model:
            if os.path.exists(self.model_file):
                self.model = load_model(self.model_file)

        if self.model is None:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(self.time_steps, self.features), return_sequences=True))
            self.model.add(LSTM(100, return_sequences=False))
            self.model.add(Dense(1))
            self.model.compile(loss='mae', optimizer='adam')

        return self.model

    def single_lstm_train(self):
        data_x, data_y = self.gen_dataset()
        train_size = int(data_x.shape[0] * 0.9)
        train_x, test_x = data_x[:train_size], data_x[train_size:]
        train_y, test_y = data_y[:train_size], data_y[train_size:]
        model = self.prepare_model()
        model.fit(train_x, train_y, epochs=200, batch_size=64, validation_data=(test_x, test_y), verbose=1,
                  shuffle=True)
        model.save(self.model_file)

    def predict_with_p230_for_stock(self, st_code):
        file = os.path.join(self.data_dir, st_code + ".csv")
        df = pd.read_csv(file, header=0, encoding='utf-8')
        if df is None:
            return None

        values = df.values[:, 1:-1].astype('float32')
        last_value = self.data_util.get_p230_redict_last_input(st_code)
        values = np.concatenate([values, last_value], axis=0)
        dataset = self.scaler.fit_transform(values)

        input = dataset[:, :-1].reshape((-1, self.time_steps, self.features))

        print(input)
        output = self.model.predict(input)
        inverse = np.concatenate([dataset[, 0:3], output, dataset[, 4:]], axis=1)
        predict = self.scaler.inverse_transform(inverse)
        return predict.flatten()[0]


    def predict_with_p230(self):
        now = datetime.datetime.now().strftime('%H:%M')
        if now < '14:40' or now > '2:50':
            return

        if self.load_p230_model() is None:
            return

        stocks = self.data_util.gen_today_p230()

        if stocks is None or len(stocks) == 0:
            return

        for stock in stocks:
            predict = self.predict_with_p230_for_stock(stock)
            if predict is None:
                continue

            print("stock = %s predict=%f" % stock, predict)

    '''
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
        #csv_write.writerow([st_code, predict])
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
    '''


if __name__ == '__main__':
    downloadClient = DownloadClient()
    downloadClient.get_today_ticks_for_stock()
    #singleLstm = SingleLstm()
    #singleLstm.single_lstm_train()
