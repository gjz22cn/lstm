import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


class LstmStockM2:
    def __init__(self):
        self.root_dir = '../'
        self.stocks_dir = os.path.join(self.root_dir, 'stocks')
        self.model_dir = os.path.join(self.root_dir, 'm_s2')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_class = 'm_s2'
        self.time_steps = 60
        self.features = 3
        self.predict_len = 1
        self.model = None
        #self.stock_list = ['002631.SZ', '002236.SZ']
        self.stock_list = ['002631.SZ']
        self.time_steps_list = [15, 30, 60, 90]

    def set_time_steps(self, time_steps):
        self.time_steps = time_steps

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
        self.model = None
        if os.path.exists(model_file):
            self.model = load_model(model_file)

        if self.model is None:
            features = self.features
            output_len = self.predict_len

            self.model = Sequential()
            self.model.add(LSTM(24, input_shape=(self.time_steps, features), return_sequences=True))
            self.model.add(LSTM(48, return_sequences=True))
            self.model.add(LSTM(96, return_sequences=True))
            self.model.add(LSTM(192, return_sequences=False))
            '''
            self.model.add(LSTM(512, input_shape=(self.time_steps, features), return_sequences=True))
            self.model.add(LSTM(512, return_sequences=False))
            '''
            self.model.add(Dense(output_len))
            self.model.compile(loss='mae', optimizer='adam')

        return self.model

    def save_stock_model(self, ts_code):
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
        file = os.path.join(self.stocks_dir, st_code + '.csv')
        cols = ['close', 'vol', 'amount']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')
        if df is None or df.shape[0] == 0:
            return None

        return df.values.astype('float32')

    def shuffle_data_poll(self, data_x, data_y):
        len = data_x.shape[0]
        index = [i for i in range(len)]
        random.shuffle(index)
        return data_x[index], data_y[index]

    def train_for_stock(self, ts_code, epochs, show_plt):
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

        if show_plt:
            data_s_x, data_s_y = data_x, data_y
        else:
            data_s_x, data_s_y = self.shuffle_data_poll(data_x, data_y)

        train_size = int(data_x.shape[0] * 0.9)
        train_x, test_x = data_s_x[:train_size], data_s_x[train_size:]
        train_y, test_y = data_s_y[:train_size], data_s_y[train_size:]

        model = self.prepare_model(ts_code)
        model.fit(train_x, train_y, epochs=epochs, batch_size=128, validation_data=(test_x, test_y), verbose=1,
                  shuffle=True)

        self.save_stock_model(ts_code)

        if show_plt:
            train_p = model.predict(train_x)
            train_inverse_input = np.concatenate([train_p, data_s[:train_size, 1:]], axis=1)
            train_p = self.scaler.inverse_transform(train_inverse_input)
            train_p = train_p[:, 0]

            test_p = model.predict(test_x)
            test_inverse_input = np.concatenate([test_p, data_s[train_size:cnt, 1:]], axis=1)
            test_p = self.scaler.inverse_transform(test_inverse_input)
            test_p = test_p[:, 0]
            #self.show_plt(data[:, 0].flatten(), train_p.flatten(), test_p.flatten(), ts_code)
            title = ts_code + '(' + str(self.time_steps) + ')'
            self.show_plt_2(data[:, 0].flatten(), train_p.flatten(), test_p.flatten(), title)

    def show_plt(self, ori, train, test, title):
        plt.figure(figsize=(30, 6))
        ori_plot = ori

        train_plot = np.empty_like(ori_plot)
        train_plot[:] = np.nan
        t_idx_s = self.time_steps
        train_plot[t_idx_s:t_idx_s + train.shape[0]] = train

        test_plot = np.empty_like(ori_plot)
        test_plot[:] = np.nan
        test_idx_s = self.time_steps + train.shape[0]
        test_plot[test_idx_s:test_idx_s + test.shape[0]] = test

        l1, = plt.plot(ori_plot, color='red', linewidth=3, linestyle=':')
        l2, = plt.plot(train_plot, color='k', linewidth=2, linestyle=':')
        l3, = plt.plot(test_plot, color='g', linewidth=2, linestyle=':')
        plt.ylabel(' ')
        plt.legend([l1, l2, l3], ('ori', 'train', 'test'), loc='best')
        plt.title(title)
        plt.show()

    def show_plt_2(self, ori, train, test, title):
        plt.figure(figsize=(30, 6))
        test_idx_s = self.time_steps + train.shape[0]

        ref = ori[test_idx_s:test_idx_s + test.shape[0]]
        delta = np.subtract(test, ref)
        delta = np.divide(delta, ref)

        l1, = plt.plot(delta, color='red', linewidth=3, linestyle=':')
        plt.ylabel(' ')
        plt.legend([l1], ('delta'), loc='best')
        plt.title(title)
        plt.show()

    def train_all_stock(self, epochs, show_plt, t_steps):
        if t_steps is None:
            for time_steps in self.time_steps_list:
                self.set_time_steps(time_steps)
                for stock in self.stock_list:
                    self.train_for_stock(stock, epochs, show_plt)
        else:
            self.set_time_steps(t_steps)
            for stock in self.stock_list:
                self.train_for_stock(stock, epochs, show_plt)

    def predict(self, ts_code):
        model = self.load_saved_model(ts_code)
        if model is None:
            return None, None

        data = self.get_dataset(ts_code)
        if data is None:
            return None, None

        data_s = self.scaler.fit_transform(data)

        idx_s = data_s.shape[0] - self.time_steps
        data_x = data_s[idx_s:]
        data_x = data_x.reshape((1, self.time_steps, data.shape[1]))

        data_y = model.predict(data_x)
        inverse_input = np.concatenate([data_y, data_s[-1:, 1:]], axis=1)
        predict = self.scaler.inverse_transform(inverse_input)
        predict = predict[:, 0].flatten()[0]

        ori_y = data[-1, 0]
        return ori_y, predict

    def predict_all_for_date(self, date):
        cols = ['ts_code', 'today', 'predict', 'chg_per']
        df = None
        for stock in self.stock_list:
            today, predict = self.predict(stock)
            if predict is None:
                continue

            chg_per = str(round(100*(predict - today)/today, 2)) + '%'
            today_str = str(round(today, 2))
            predict_str = str(round(predict, 2))
            df_i = pd.DataFrame([[stock, today_str, predict_str, chg_per]], columns=cols)
            if df is None:
                df = df_i
            else:
                df = pd.concat((df, df_i), axis=0)

        file = os.path.join(self.model_dir, 'p_'+date+'_'+str(self.time_steps)+'.csv')
        df.to_csv(file, mode='w', header=True, float_format='%.2f', encoding="utf_8_sig")

        print(date+'('+str(self.time_steps)+'):')
        print(df)

    def predict_for_all_model(self, date):
        df = None
        for i in range(len(self.time_steps_list)):
            self.set_time_steps(self.time_steps_list[i])
            if i == 0:
                cols = ['ts_code', 'today', 'chg_'+str(self.time_steps_list[i])]
            else:
                cols = ['chg_'+str(self.time_steps_list[i])]

            df_per_model = None
            for j in range(len(self.stock_list)):
                today, predict = self.predict(self.stock_list[j])
                if predict is None:
                    continue

                chg_per = str(round(100 * (predict - today) / today, 2)) + '%'
                today_str = str(round(today, 2))
                if j == 0:
                    if i == 0:
                        df_per_model = pd.DataFrame([[self.stock_list[j], today_str, chg_per]], columns=cols)
                    else:
                        df_per_model = pd.DataFrame([[chg_per]], columns=cols)
                else:
                    if i == 0:
                        df_j = pd.DataFrame([[self.stock_list[j], today_str, chg_per]], columns=cols)
                    else:
                        df_j = pd.DataFrame([[chg_per]], columns=cols)
                    df_per_model = pd.concat((df_per_model, df_j), axis=0)

            df_per_model = df_per_model.reset_index(drop=True)
            if i == 0:
                df = df_per_model
            else:
                df = pd.concat((df, df_per_model), axis=1)

        file = os.path.join(self.model_dir, 'p_all_' + date + '.csv')
        df.to_csv(file, mode='w', header=True, float_format='%.2f', encoding="utf_8_sig")
        print(date)
        print(df)


if __name__ == '__main__':
    lstmStockM2 = LstmStockM2()
    # time_steps: 15, 30, 60, 90
    lstmStockM2.train_all_stock(200, 1, 30)


    #lstmStockM2.set_time_steps(60)
    #lstmZhiShu.train_for_zhishu('000001.SH', 100, 1)
    #lstmZhiShu.predict_all_for_date('20190125')
    #lstmStockM2.predict_for_all_model('20190129')