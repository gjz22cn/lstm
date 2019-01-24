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
        self.zhishu_list = ['000001.SH', '000016.SH', '000300.SH',
                            '000905.SH', '399001.SZ', '399005.SZ',
                            '399006.SZ', '399016.SZ']

    def load_saved_model(self, ts_code):
        st_code = ts_code.split('.')[0]
        model_file = os.path.join(self.model_dir, st_code + '_' + str(self.time_steps) + '.h5')
        print(model_file)
        if os.path.exists(model_file):
            self.model = load_model(model_file)
            return self.model

        return None

    def prepare_model(self, ts_code):
        st_code = ts_code.split('.')[0]
        model_file = os.path.join(self.model_dir, st_code + '_' + str(self.time_steps) + '.h5')
        if os.path.exists(model_file):
            self.model = load_model(model_file)

        if self.model is None:
            features = self.features
            output_len = self.predict_len
            if ts_code == 'm2':
                features = self.features * len(self.zhishu_list)
                output_len = self.predict_len * len(self.zhishu_list)

            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(self.time_steps, features), return_sequences=True))
            self.model.add(LSTM(100, return_sequences=False))
            self.model.add(Dense(output_len))
            self.model.compile(loss='mae', optimizer='adam')

        return self.model

    def save_zhishu_model(self, ts_code):
        st_code = ts_code.split('.')[0]
        model_file = os.path.join(self.model_dir, st_code + '_' + str(self.time_steps) + '.h5')
        self.model.save(model_file)

    def gen_dataset(self, data):
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
        if df is None or df.shape[0] == 0:
            return None

        return df.values.astype('float32')

    def shuffle_data_poll(self, data_x, data_y):
        return data_x, data_y
        '''
        len = data_x.shape[0]
        index = [i for i in range(len)]
        random.shuffle(index)
        return  data_x[index], data_y[index]
        '''

    def train_for_zhishu(self, ts_code, epochs):
        show_plt = False
        data = self.get_dataset(ts_code)
        if data is None:
            return

        cnt = data.shape[0] - self.time_steps - (self.predict_len-1)
        if cnt < 1:
            return

        data_s = self.scaler.fit_transform(data)
        data_x, data_y = self.gen_dataset(data_s)
        if data_x is None or data_y is None:
            return

        data_s_x, data_s_y = self.shuffle_data_poll(data_x, data_y)
        train_size = int(data_x.shape[0] * 0.9)
        train_x, test_x = data_s_x[:train_size], data_s_x[train_size:]
        train_y, test_y = data_s_y[:train_size], data_s_y[train_size:]

        model = self.prepare_model(ts_code)
        model.fit(train_x, train_y, epochs=epochs, batch_size=64, validation_data=(test_x, test_y), verbose=1,
                  shuffle=True)

        self.save_zhishu_model(ts_code)

        if show_plt:
            train_p = model.predict(train_x)
            train_inverse_input = np.concatenate([train_p, data_s[:train_size, 1:]], axis=1)
            train_p = self.scaler.inverse_transform(train_inverse_input)
            train_p = train_p[:, 0]

            test_p = model.predict(test_x)
            test_inverse_input = np.concatenate([test_p, data_s[train_size:cnt, 1:]], axis=1)
            test_p = self.scaler.inverse_transform(test_inverse_input)
            test_p = test_p[:, 0]
            self.show_plt(data[:, 0].flatten(), train_p.flatten(), test_p.flatten())

    def show_plt(self, ori, train, test):
        plt.figure(figsize=(20, 6))
        ori_plot = ori

        train_plot = np.empty_like(ori_plot)
        train_plot[:] = np.nan
        t_idx_s = self.time_steps
        train_plot[t_idx_s:t_idx_s + train.shape[0]] = train

        test_plot = np.empty_like(ori_plot)
        test_plot[:] = np.nan
        test_idx_s = self.time_steps + train.shape[0]
        test_plot[test_idx_s:test_idx_s + test.shape[0]] = test

        l1, = plt.plot(ori_plot, color='red', linewidth=3, linestyle='--')
        l2, = plt.plot(train_plot, color='k', linewidth=2, linestyle='--')
        l3, = plt.plot(test_plot, color='g', linewidth=2, linestyle='--')
        plt.ylabel('yuan')
        plt.legend([l1, l2, l3], ('ori', 'train', 'test'), loc='best')
        plt.title('Prediction')
        plt.show()

    def train_all_zhishu(self, epochs):
        for zhishu in self.zhishu_list:
            self.train_for_zhishu(zhishu, epochs)

    def predict(self, ts_code):
        model = self.load_saved_model(ts_code)
        if model is None:
            return None, None, None

        data = self.get_dataset(ts_code)
        if data is None:
            return None, None, None

        data_s = self.scaler.fit_transform(data)

        idx_s = data_s.shape[0] - self.time_steps
        data_x = data_s[idx_s:]
        data_x = data_x.reshape((1, self.time_steps, data.shape[1]))

        data_y = model.predict(data_x)
        inverse_input = np.concatenate([data_y, data_s[-1:, 1:]], axis=1)
        predict = self.scaler.inverse_transform(inverse_input)
        predict = predict[:, 0].flatten()[0]

        ori_y = data[-1, 0]
        return ts_code, ori_y, predict

    def predict_all_for_date(self, date):
        cols = ['ts_code', 'today', 'predict', 'chg_per']
        df = None
        for zhishu in self.zhishu_list:
            zhishu, today, predict = self.predict(zhishu)
            if zhishu is None:
                continue

            chg_per = str(round(100*(predict - today)/today, 2)) + '%'
            today_str = str(round(today, 2))
            predict_str = str(round(predict, 2))
            df_i = pd.DataFrame([[zhishu, today_str, predict_str, chg_per]], columns=cols)
            if df is None:
                df = df_i
            else:
                df = pd.concat((df, df_i), axis=0)

        file = os.path.join(self.model_dir, 'p_'+date+'.csv')
        df.to_csv(file, mode='w', header=True, float_format='%.2f', encoding="utf_8_sig")

        print(date)
        print(df)

    ######################################################################################
    # codes for model 2
    ######################################################################################
    def gen_dataset_for_m2(self, data):
        cnt = data.shape[0] - self.time_steps - (self.predict_len-1)
        data_x = data[:cnt]
        for i in range(1, self.time_steps):
            data_x = np.concatenate([data_x, data[i:i + cnt]], axis=1)

        print(data.shape)

        data_x = data_x.reshape((cnt, self.time_steps, data.shape[1]))
        col_s = self.time_steps
        data_y = data[col_s:col_s+cnt, 0:1]
        for i in range(1, len(self.zhishu_list)):
            data_y = np.concatenate([data_y, data[col_s:col_s+cnt, (3*i):(3*i+1)]], axis=1)

        for i in range(1, self.predict_len):
            for i in range(1, len(self.zhishu_list)):
                data_y = np.concatenate([data_y, data[col_s+i:col_s+i+cnt, (3*i):(3*i+1)]], axis=1)

        data_y = data_y.reshape((cnt, len(self.zhishu_list)*self.predict_len))

        return data_x, data_y

    def train_for_m2(self, epochs):
        file = os.path.join(self.zhishus_dir, "all_zhishu.csv")
        df = pd.read_csv(file, header=0, encoding='utf-8')
        if df is None:
            return

        cnt = df.shape[0] - self.time_steps - (self.predict_len-1)
        if cnt < 1:
            return

        data = df.values[:, 2:].astype('float32')
        data_s = self.scaler.fit_transform(data)
        data_x, data_y = self.gen_dataset_for_m2(data_s)
        if data_x is None or data_y is None:
            return

        data_s_x, data_s_y = self.shuffle_data_poll(data_x, data_y)
        train_size = int(data_x.shape[0] * 0.9)
        train_x, test_x = data_s_x[:train_size], data_s_x[train_size:]
        train_y, test_y = data_s_y[:train_size], data_s_y[train_size:]

        model = self.prepare_model('m2')
        model.fit(train_x, train_y, epochs=epochs, batch_size=64, validation_data=(test_x, test_y), verbose=1,
                  shuffle=True)

        self.save_zhishu_model('m2')

        show_plt = True
        if show_plt:
            train_p = model.predict(train_x)
            inv_in1 = np.concatenate([train_p[:, 0:1], data_s[:train_size, 1:3]], axis=1)
            for i in range(1, len(self.zhishu_list)):
                inv_in1 = np.concatenate([inv_in1, train_p[:, i:i+1]], axis=1)
                inv_in1 = np.concatenate([inv_in1, data_s[:train_size, 3*i+1:3*i+3]], axis=1)

            train_p = self.scaler.inverse_transform(inv_in1)

            test_p = model.predict(test_x)
            inv_in2 = np.concatenate([test_p[:, 0:1], data_s[self.time_steps+train_size:, 1:3]], axis=1)
            for i in range(1, len(self.zhishu_list)):
                inv_in2 = np.concatenate([inv_in2, test_p[:, i:i + 1]], axis=1)
                inv_in2 = np.concatenate([inv_in2, data_s[self.time_steps+train_size:, 3 * i + 1:3 * i + 3]], axis=1)

            test_p = self.scaler.inverse_transform(inv_in2)

            for i in range(0, len(self.zhishu_list)):
                self.show_plt(data[:, 3*i].flatten(), train_p[:, 3*i].flatten(), test_p[:, 3*i].flatten())


if __name__ == '__main__':
    lstmZhiShu = LstmZhiShu()
    #lstmZhiShu.train_all_zhishu(10)
    #lstmZhiShu.train_for_zhishu('000016.SH', 200)
    #lstmZhiShu.predict_all_for_date('20190124')
    lstmZhiShu.train_for_m2(10)