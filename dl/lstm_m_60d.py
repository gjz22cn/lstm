import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from pre_process import DataUtil
from dingding_msg import DingdingMsg


class LstmM60d:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_class = 'm_s1'
        self.root_dir = '../'
        self.model_dir = os.path.join(self.root_dir, self.model_class)
        self.stocks_dir = os.path.join(self.root_dir, 'stocks')
        self.predict_dir = os.path.join(self.model_dir, 'predict')
        self.features = 6
        self.predict_len = 3
        self.time_steps = 15
        self.model_name = self.gen_model_name()
        self.model_file = os.path.join(self.model_dir, self.model_name+'.h5')
        self.skip_saved_model = False
        self.model = None
        self.data_util = DataUtil('../')
        self.dingding = DingdingMsg()
        self.stocks = []
        self.stocks_num = 0
        self.stock_idx = 0
        self.batch_size = 128
        self.train_pool_x = []
        self.train_pool_y = []
        self.data_idx = 0
        self.train_pool_size = 0
        self.steps_per_epoch = 0
        self.pool_p = 200

    def gen_model_name(self):
        return self.model_class + '_' + str(self.time_steps) + '_' + str(self.predict_len)

    def set_time_steps(self, time_steps):
        self.time_steps = time_steps
        self.model_name = self.gen_model_name()
        self.model_file = os.path.join(self.model_dir, self.model_name + '.h5')
        if self.time_steps == 15:
            self.pool_p = 200
        elif self.time_steps == 30:
            self.pool_p = 100
        elif self.time_steps == 60:
            self.pool_p = 50
        elif self.time_steps == 90:
            self.pool_p = 25
        elif self.time_steps == 180:
            self.pool_p = 20
        elif self.time_steps == 270:
            self.pool_p = 15

    def get_predict_stock_list(self):
        file = os.path.join(self.model_dir, 'stock_list.csv')
        df = pd.read_csv(file, header=0, usecols=['ts_code'], encoding='utf-8')
        return df.values.flatten()

    def load_saved_model(self):
        print(self.model_file)
        if os.path.exists(self.model_file):
            self.model = load_model(self.model_file)
            return self.model

        return None

    def prepare_model(self):
        if os.path.exists(self.model_file):
            self.model = load_model(self.model_file)

        if self.model is None:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(self.time_steps, self.features), return_sequences=True))
            self.model.add(LSTM(100, return_sequences=False))
            self.model.add(Dense(self.predict_len))
            self.model.compile(loss='mae', optimizer='adam')

        return self.model

    def get_stocks(self):
        #self.stocks = self.data_util.get_valid_single_stock_list()
        self.stocks = self.data_util.get_all_stocks(3)
        random.shuffle(self.stocks)
        if self.stocks is None:
            return
        self.stocks_num = len(self.stocks)

    def gen_dataset(self, stock_data):
        data = self.scaler.fit_transform(stock_data)
        cnt = data.shape[0] - self.time_steps - (self.predict_len-1)
        data_x = data[:cnt]
        for i in range(1, self.time_steps):
            data_x = np.concatenate([data_x, data[i:i + cnt]], axis=1)

        data_x = data_x.reshape((cnt, self.time_steps, data.shape[1]))
        data_y = data[self.time_steps:self.time_steps+cnt, 3:4]
        for i in range(1, self.predict_len):
            data_y = np.concatenate([data_y, data[self.time_steps+i:self.time_steps+i+cnt, 3:4]], axis=1)
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

        ava_data_cnt_delta = data.shape[0] - self.time_steps - (self.predict_len - 1)
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

    def shuffle_train_poll(self):
        index = [i for i in range(self.data_idx, self.data_idx+self.train_pool_size)]
        random.shuffle(index)
        self.train_pool_x = self.train_pool_x[index]
        self.train_pool_y = self.train_pool_y[index]

    def generator(self):
        while True:
            while self.train_pool_size < 1024*self.pool_p:
                if self.stock_idx == self.stocks_num:
                    break

                self.load_next_stock()
                self.shuffle_train_poll()

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
        ava = count - self.time_steps - (self.predict_len-1)
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
                            shuffle=False,
                            initial_epoch=0)

        model.save(self.model_file)

    def predict_for_stock(self, st_code):
        file = os.path.join(self.stocks_dir, st_code + '.csv')
        data = pd.read_csv(file, header=0, usecols=['open', 'high', 'low', 'close', 'vol', 'amount'], encoding='utf-8')
        if data is None:
            return None

        if data.shape[0] < self.time_steps:
            return None

        values = data.values.astype('float32')
        data_s = self.scaler.fit_transform(values)
        input = data_s[0-self.time_steps:].reshape((1, self.time_steps, data_s.shape[1]))
        output = self.model.predict(input)
        start_row = 0 - self.predict_len
        inverse = np.concatenate([data_s[start_row:, 0:3], output.reshape((-1, 1)), data_s[start_row:, 4:]], axis=1)
        predict = self.scaler.inverse_transform(inverse)
        return predict[:, 3].flatten()

    def get_date_idx(self, df, date):
        for row in df.itertuples():
            if date == row[1]:
                return row[0]

        return -1

    def predict_stock_for_date(self, st_code, date):
        file = os.path.join(self.stocks_dir, st_code + '.csv')
        cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
        df = pd.read_csv(file, header=0, usecols=cols, dtype={'trade_date': str}, encoding='utf-8')
        if df is None:
            return None
        df = df.reset_index(drop=True)

        row = self.get_date_idx(df, date)
        if row == -1:
            return None

        if row < self.time_steps:
            return None

        df = df[row-self.time_steps+1:row+1]
        values = df.values[:, 1:].astype('float32')
        #print(st_code, values)
        data_s = self.scaler.fit_transform(values)
        input = data_s[0:].reshape((1, self.time_steps, data_s.shape[1]))
        output = self.model.predict(input)
        start_row = 0 - self.predict_len
        inverse = np.concatenate([data_s[start_row:, 0:3], output.reshape((-1, 1)), data_s[start_row:, 4:]], axis=1)
        predict = self.scaler.inverse_transform(inverse)
        return predict[:, 3].flatten()

    def do_predict(self):
        model = self.load_saved_model()
        if model is None:
            return

        stocks = ['600363', '600820', '002850', '002500', '300098', '002465', '002300']
        msg = 'From Robot0(' + self.model_name + '):\n'
        for stock in stocks:
            predict = self.predict_for_stock(stock)
            if predict is None:
                continue

            msg = msg + stock + ":"
            for item in predict:
                msg = msg + str(round(item, 2)) + ","

            msg = msg + str(round((predict[-1]-predict[0])/predict[0], 3)) + '\n'

        print(msg)
        self.dingding.send_msg(msg)

    def get_top_stocks(self):
        model = self.load_saved_model()
        if model is None:
            return

        stocks = self.data_util.get_valid_single_stock_list()
        result = []

        for stock in stocks:
            st_code = stock.split('.')[0]
            predict = self.predict_for_stock(st_code)
            if predict is None:
                continue

            item_r = []
            item_r.append(st_code)
            for p in predict:
                item_r.append(round(p, 2))

            item_r.append(round((predict[-1]-predict[0])/predict[0], 3))
            result.append(item_r)

        result = sorted(result, key=lambda item: item[self.predict_len+1], reverse=True)

        msg = 'From Robot1(' + self.model_name + '):\n'
        for i in range(10):
            msg = msg + result[i][0] + ':'
            for j in range(1, self.predict_len+1):
                msg = msg + str(result[i][j]) + ","
            msg = msg + str(result[i][self.predict_len+1]) + '\n'

        print(msg)
        print(result[:10])
        #self.dingding.send_msg(msg)

    def predict_for_date(self, date):
        model = self.load_saved_model()
        if model is None:
            print("model is None!")
            return

        valid_stocks = []

        stocks = self.get_predict_stock_list()
        #stocks = self.data_util.get_all_stocks(3)
        result = []
        for stock in stocks:
            st_code = stock.split('.')[0]
            predict = self.predict_stock_for_date(st_code, date)
            if predict is None:
                continue

            item_r = []
            item_r.append(st_code)
            for p in predict:
                item_r.append(p)

            item_r.append((predict[-1] - predict[0]) / predict[0])
            result.append(item_r)
            valid_stocks.append(stock)

        #result = sorted(result, key=lambda item: item[self.predict_len + 1], reverse=True)

        cols = ['stock', 'd1', 'd2', 'd3', 'per']
        df = pd.DataFrame(result, columns=cols)
        dir = os.path.join(self.predict_dir, self.model_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = os.path.join(dir, date + '.csv')
        df.to_csv(file, columns=cols, mode='w', header=True, float_format='%.3f', encoding="utf_8_sig")

        df_stocks = pd.DataFrame(valid_stocks, columns=['ts_code'])
        file = os.path.join(self.model_dir, 'stock_list.csv')
        df_stocks.to_csv(file, columns=['ts_code'], mode='w', header=True, encoding="utf_8_sig")

    def predict_for_all_model_by_date(self, date):
        time_steps_array = [90, 60, 30, 15]
        for time_steps in time_steps_array:
            self.set_time_steps(time_steps)
            self.predict_for_date(date)

    def get_predict(self, date, time_steps):
        model_name = self.model_class + '_' + str(time_steps) + '_' + str(self.predict_len)
        dir = os.path.join(self.predict_dir, model_name)
        file = os.path.join(dir, date + '.csv')
        cols = ['stock', 'd1', 'd2', 'd3', 'per']
        df = pd.read_csv(file, header=0, usecols=cols, dtype={'stock': str}, encoding='utf-8')
        return df

    def get_stock_w(self, df):
        df = df.sort_values(by='per', ascending=False)
        df = df.reset_index(drop=True)
        v = [i for i in range(1, df.shape[0]+1)]
        df_v = pd.DataFrame(np.reshape(v, (-1, 1)), columns=['w'])
        df = pd.concat([df, df_v], axis=1)
        df = df.sort_values(by='stock', ascending=True)
        df = df.reset_index(drop=True)
        return df.values[:, -1].flatten()

    def sort_for_predict(self, date):
        df_15 = self.get_predict(date, 15)
        w_15 = self.get_stock_w(df_15)

        df_30 = self.get_predict(date, 30)
        w_30 = self.get_stock_w(df_30)

        df_60 = self.get_predict(date, 60)
        w_60 = self.get_stock_w(df_60)

        df_90 = self.get_predict(date, 90)
        w_90 = self.get_stock_w(df_90)

        #print(w_15.shape, w_30.shape, w_60.shape)
        w2 = np.add(w_15, w_30)
        w3 = np.add(w2, w_60)
        w4 = np.add(w3, w_90)
        df_w_15 = pd.DataFrame(np.reshape(w_15, (-1, 1)), columns=['w_15'])
        df_w_30 = pd.DataFrame(np.reshape(w_30, (-1, 1)), columns=['w_30'])
        df_w_60 = pd.DataFrame(np.reshape(w_60, (-1, 1)), columns=['w_60'])
        df_w_90 = pd.DataFrame(np.reshape(w_90, (-1, 1)), columns=['w_90'])
        df_w2 = pd.DataFrame(np.reshape(w2, (-1, 1)), columns=['w2'])
        df_w3 = pd.DataFrame(np.reshape(w3, (-1, 1)), columns=['w3'])
        df_w4 = pd.DataFrame(np.reshape(w4, (-1, 1)), columns=['w4'])
        df = df_15.sort_values(by='stock', ascending=True)
        df = df.reset_index(drop=True)
        df = pd.concat([df, df_w_15, df_w_30, df_w_60, df_w_90, df_w2, df_w3, df_w4], axis=1)

        df_2 = df.sort_values(by='w2', ascending=True)
        df_2 = df_2.reset_index(drop=True)
        df_3 = df.sort_values(by='w3', ascending=True)
        df_3 = df_3.reset_index(drop=True)
        df_4 = df.sort_values(by='w4', ascending=True)
        df_4 = df_4.reset_index(drop=True)

        df_predict = pd.concat([df_2[:2], df_3[:2], df_4[:2]], axis=0)

        print(date)
        print(df_predict)
        file = os.path.join(self.predict_dir, date + '.csv')
        df_predict.to_csv(file, mode='w', header=True, float_format='%.3f', encoding="utf_8_sig")

    def evaluate_model(self, time_steps):
        # load model
        self.set_time_steps(time_steps)
        model = self.load_saved_model()
        if model is None:
            print("model is None!")
            return

        stocks = self.get_predict_stock_list()
        # stocks = self.data_util.get_all_stocks(3)
        result = []
        for stock in stocks:
            # get dataset
            st_code = stock.split('.')[0]
            file = os.path.join(self.stocks_dir, st_code + '.csv')
            df = pd.read_csv(file, header=0, usecols=['open', 'high', 'low', 'close', 'vol', 'amount'], encoding='utf-8')
            if df is None:
                continue

            cnt = df.shape[0] - self.time_steps - (self.predict_len - 1)
            if cnt < 1:
                continue

            data = df.values.astype('float32')
            data_x, data_y = self.gen_dataset(data)

            # predict
            output = model.predict(data_x)
            predict = []
            for i in range(self.predict_len):
                inverse = np.concatenate([data[:cnt, 0:3], output[:, i:i+1], data[:cnt, 4:]], axis=1)
                if i == 0:
                    predict = self.scaler.inverse_transform(inverse)[:, 3:4]
                else:
                    predict = np.concatenate([predict, self.scaler.inverse_transform(inverse)[:, 3:4]], axis=1)

            # calc result
            ori_y = data[self.time_steps:self.time_steps+cnt, 3:4]
            for i in range(1, self.predict_len):
                ori_y = np.concatenate([ori_y, data[self.time_steps+i:self.time_steps+i+cnt, 3:4]], axis=1)

            r1 = np.fabs(np.subtract(ori_y, predict))
            r2 = np.true_divide(r1, ori_y)
            max_v = r2.max(axis=0)
            min_v = r2.min(axis=0)
            avg_v = r2.mean(axis=0)
            r_item = [st_code]
            for i in range(self.predict_len):
                r_item.append(min_v[i])
                r_item.append(max_v[i])
                r_item.append(avg_v[i])

            result.append(r_item)

        cols = ['st_code']
        for i in range(self.predict_len):
            cols.append('min'+str(i+1))
            cols.append('max'+str(i+1))
            cols.append('avg'+str(i+1))

        df_stocks = pd.DataFrame(result, columns=cols)
        file = os.path.join(self.model_dir, 'eva_' + self.model_class + '_' + str(self.time_steps) + '.csv')
        df_stocks.to_csv(file, mode='w', header=True, float_format='%.3f', encoding="utf_8_sig")


if __name__ == '__main__':
    lstmM60d = LstmM60d()
    # #################################
    # Train
    #   time_steps: 15, 30, 60, 90, 270
    # ##################################
    #lstmM60d.set_time_steps(90)
    #lstmM60d.train()

    # #################################
    # Evaluate
    #   time_steps: 15, 30, 60, 90, 270
    # ##################################
    lstmM60d.evaluate_model(15)
    lstmM60d.evaluate_model(30)
    lstmM60d.evaluate_model(60)
    lstmM60d.evaluate_model(90)

    #lstmM60d.do_predict()
    #lstmM60d.get_top_stocks()

    '''
    lstmM60d.predict_for_all_model_by_date('20190107')
    lstmM60d.predict_for_all_model_by_date('20190108')
    lstmM60d.predict_for_all_model_by_date('20190109')
    lstmM60d.predict_for_all_model_by_date('20190110')
    lstmM60d.predict_for_all_model_by_date('20190111')
    lstmM60d.predict_for_all_model_by_date('20190115')
    lstmM60d.predict_for_all_model_by_date('20190116')
    lstmM60d.predict_for_all_model_by_date('20190117')
    lstmM60d.predict_for_all_model_by_date('20190118')
    
    lstmM60d.sort_for_predict('20190107')
    lstmM60d.sort_for_predict('20190108')
    lstmM60d.sort_for_predict('20190109')
    lstmM60d.sort_for_predict('20190110')
    lstmM60d.sort_for_predict('20190111')
    lstmM60d.sort_for_predict('20190115')
    lstmM60d.sort_for_predict('20190116')
    lstmM60d.sort_for_predict('20190117')
    lstmM60d.sort_for_predict('20190118')
    '''

    #lstmM60d.predict_for_all_model_by_date('20190121')
    #lstmM60d.sort_for_predict('20190121')