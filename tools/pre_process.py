import os
import numpy as np
import pandas as pd
import tushare as ts
from DownloadClient import DownloadClient
import datetime


# df = ts.get_tick_data('002001', date='2018-11-29', src='tt')
# print(df)

class DataUtil:
    def __init__(self, dir):
        self.root_dir = dir
        self.start_date = '20140101'
        self.end_date = '20181130'
        self.stocks_dir = os.path.join(self.root_dir, 'stocks')
        self.stock_dir = os.path.join(self.root_dir, 'stock')
        self.stock_today_dir = os.path.join(self.root_dir, 'stock')
        self.p230_dir = os.path.join(self.root_dir, 'p230')
        self.single_data_dir = os.path.join(self.root_dir, 'single_d')
        ts.set_token('b1de6890364825a4b7b2d227b64c09a486239daf67451c5638404c62')
        self.pro = ts.pro_api()
        self.downloadClient = DownloadClient()
        self.step_len = 10

    def download_stock(self, code, name):
        df = self.pro.daily(ts_code=code, start_date=self.start_date, end_date=self.end_date)
        '''
        名称	        类型	    描述
        ts_code	    str	    股票代码
        trade_date	str	    交易日期
        open	    float	开盘价
        high	    float	最高价
        low	        float	最低价
        close	    float	收盘价
        pre_close	float	昨收价
        change	    float	涨跌额
        pct_chg	    float	涨跌幅
        vol	        float	成交量 （手）
        amount	    float	成交额 （千元）
        '''
        file_path = os.path.join(self.stock_dir, code + '.csv')
        mode = 'w'
        need_header = True
        if os.path.exists(file_path):
            mode = 'a'
            need_header = False

        # colunms 保存指定的列索引
        columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol',
                   'amount']

        # df = df.sort_values(by='trade_date', ascending=True)

        # df.to_csv(file_path, columns=columns, mode=mode, header=need_header, encoding="utf_8_sig")

    def get_all_stocks(self, type):
        file = os.path.join(self.root_dir, 'stock_list.csv')
        if not os.path.exists(file):
            return []

        data = pd.read_csv(file, header=0, usecols=['ts_code'], encoding='utf-8')
        data = data.values.flatten()

        output = []
        if type == 1:
            for v in data:
                if v.startswith('002') or v.startswith('300'):
                    output.append(v)
        elif type == 2:
            for v in data:
                if v.startswith('002') or v.startswith('300'):
                    continue
                output.append(v)
        else:
            output = data

        return output

    def get_p230_info_for_stock(self, row):
        print(row)
        st_code = row['ts_code'].split('.')[0]
        df = ts.get_tick_data(st_code, date=row['trade_date'], src='tt')
        df = df.sort_values(by='time', ascending=True)
        print(df)
        return 1

    def collect_data_for_stock(self, st_code):
        df = self.pro.daily(ts_code=st_code, start_date=self.start_date, end_date=self.end_date)
        df['p230'] = np.nan
        for index, row in df.iterrows():
            p230 = self.get_p230_info_for_stock(row)
            break

    def collect_data_for_stocks(self):
        stocks = self.get_all_stocks(1)
        for stock in stocks:
            self.collect_data_for_stock(stock)
            break

    def get_p230_in_date_file(self, date_file):
        if not date_file.endswith('.csv'):
            return None, None, None, None, None, None, None

        cols = ['time', 'price', 'change', 'volume', 'amount', 'type']
        df = pd.read_csv(date_file, header=0, usecols=cols, encoding='utf-8')

        for i in range(df.shape[0]-1, -1, -1):
            if df[i:i+1]['time'].values[0] <= "14:40:00":
                break

        if i == 0:
            return None, None, None, None, None, None, None

        open_p = df[0:1]['price'].values[0]
        close = df[i:i+1]['price'].values[0]
        p230 = close
        df = df[:i + 1]
        high = df['price'].max()
        low = df['price'].min()
        vol = df['volume'].sum()
        amount = df['amount'].sum()/1000

        return open_p, high, low, close, vol, amount, p230

    def gen_p230_for_stock(self, ts_code):
        st_code = ts_code.split('.')[0]
        dir = os.path.join(self.stock_dir, st_code)
        if not os.path.exists(dir):
            return

        data = []
        list_fenbi = os.listdir(dir)
        list_fenbi = sorted(list_fenbi)
        for item in list_fenbi:
            if len(item) < 19:
                continue
            date = item[7:15]
            open_p, high, low, close, vol, amount, p230 = self.get_p230_in_date_file(os.path.join(dir, item))

            if p230 is None:
                continue

            data.append([date, open_p, high, low, close, vol, amount, p230])

        cols = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df_p230 = pd.DataFrame(data, columns=cols)
        p230_path = os.path.join(self.p230_dir, st_code + '.csv')
        df_p230.to_csv(p230_path, columns=cols, mode='w', header=True, encoding="utf_8_sig")

    def gen_p230_for_stocks(self):
        stocks = self.get_all_stocks(1)
        for stock in stocks:
            self.gen_p230_for_stock(stock)
            break

    def is_valid_single_stock(self, ts_code):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.p230_dir, st_code + '.csv')
        if not os.path.exists(file):
            return False

        date_array = ['20181203', '20181204', '20181205', '20181206', '20181207',
                      '20181210', '20181211', '20181212', '20181213', '20181214',
                      '20181217', '20181218', '20181219', '20181220', '20181221',
                      '20181224', '20181225', '20181226', '20181227', '20181228']

        df = pd.read_csv(file, header=0, usecols=['date'], dtype={'date': str}, encoding='utf-8')
        date_a2 = df.values.flatten()

        return self.is_date_list_valid(date_a2)


    def gen_single_stock_list(self):
        stocks = self.get_all_stocks(1)
        single_stocks = []
        for stock in stocks:
            if not self.is_valid_single_stock(stock):
                continue
            single_stocks.append(stock)

        if len(single_stocks) == 0:
            return

        df = pd.DataFrame(single_stocks, columns=['ts_code'])
        file_path = os.path.join(self.single_data_dir, 'single_stock_list.csv')
        df.to_csv(file_path, columns=['ts_code'], mode='w', header=True, encoding="utf_8_sig")

    def get_valid_single_stock_list(self):
        file_path = os.path.join(self.single_data_dir, 'single_stock_list.csv')
        df = pd.read_csv(file_path, header=0, usecols=['ts_code'], encoding='utf-8')
        return df.values.flatten()

    def is_date_list_valid(self, date):
        date_array = ['20181203', '20181204', '20181205', '20181206', '20181207',
                      '20181210', '20181211', '20181212', '20181213', '20181214',
                      '20181217', '20181218', '20181219', '20181220', '20181221',
                      '20181224', '20181225', '20181226', '20181227', '20181228']

        if len(date_array) != len(date):
            return False

        for i in range(len(date_array)):
            if date_array[i] != date[i]:
                return False

        return True

    def gen_single_data_for_stock(self, ts_code):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.stocks_dir, st_code+'.csv')
        p230_file = os.path.join(self.p230_dir, st_code + '.csv')

        if not os.path.exists(file):
            print("gen_single_dataset_for_stock(), no stock file for %s" % ts_code)
            return False

        if not os.path.exists(p230_file):
            print("gen_single_dataset_for_stock(), no p230 file for %s" % ts_code)
            return False

        cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
        df = pd.read_csv(file, header=0, usecols=cols, dtype={'trade_date': str}, encoding='utf-8')
        if df is None:
            return False

        row_n = df.shape[0]

        if row_n < 20:
            print("gen_single_dataset_for_stock(), stock file's data is not enough for %s" % ts_code)
            return False

        values = df.iloc[-20:].values

        if not self.is_date_list_valid(values[:, 0:1].flatten()):
            print("gen_single_dataset_for_stock(), stock file's date list is invalid for %s" % ts_code)
            return False

        df_p230 = pd.read_csv(p230_file, header=0, usecols=['date', 'p230'], dtype={'date': str}, encoding='utf-8')
        if df_p230 is None:
            print("gen_single_dataset_for_stock(), p230 file for %s has no data" % ts_code)
            return False

        p230_values = df_p230.values
        if not self.is_date_list_valid(p230_values[:, 0:1].flatten()):
            print("gen_single_dataset_for_stock(), p230 file's date list is invalid for %s" % ts_code)
            return False

        new_values = np.concatenate([values, p230_values[:, 1:]], axis=1)
        new_cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        new_df = pd.DataFrame(new_values, columns=new_cols)
        file_path = os.path.join(self.single_data_dir, 's_'+st_code+'.csv')
        new_df.to_csv(file_path, columns=new_cols, mode='w', header=True, encoding="utf_8_sig")

        return True

    def gen_single_data(self):
        stocks = self.get_valid_single_stock_list()
        new_stocks = []
        need_save = False
        for i in range(len(stocks)):
            if self.gen_single_data_for_stock(stocks[i]):
                new_stocks.append(stocks[i])
            else:
                need_save = True

        print("new_stocks:", new_stocks)

        if need_save:
            df = pd.DataFrame(new_stocks, columns=['ts_code'])
            file_path = os.path.join(self.single_data_dir, 'single_stock_list.csv')
            df.to_csv(file_path, columns=['ts_code'], mode='w', header=True, encoding="utf_8_sig")

    def get_single_dataset(self):
        stocks = self.get_valid_single_stock_list()
        for stock in stocks:
            self.get_single_dataset_for_stock(stock)

    def gen_train_data_for_single_by_stock(self, ts_code, step_len):
        train_data_dir = self.single_data_dir + '_' + str(step_len)
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.single_data_dir, 's_'+st_code + '.csv')
        file_train = os.path.join(train_data_dir, st_code+'.csv')
        cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')

        file_230 = os.path.join(self.p230_dir, st_code + '.csv')
        cols_230 = ['open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df_230 = pd.read_csv(file_230, header=0, usecols=cols_230, encoding='utf-8')
        df_230.rename(columns={'open': 'open_l',
                               'high': 'high_l',
                               'low': 'low_l',
                               'close': 'close_l',
                               'vol': 'vol_l',
                               'amount': 'amount_l',
                               'p230': 'p230_l'}, inplace=True)

        result = df[['close']]
        result.columns = ['result']

        new_df = df

        for i in range(1, step_len-1):
            temp_df = df.shift(0-i)
            temp_df.rename(columns={'open': 'open' + '_' + str(i),
                                    'high': 'high' + '_' + str(i),
                                    'low': 'low' + '_' + str(i),
                                    'close': 'close' + '_' + str(i),
                                    'vol': 'vol' + '_' + str(i),
                                    'amount': 'amount' + '_' + str(i),
                                    'p230': 'p230' + '_' + str(i)}, inplace=True)

            new_df = pd.concat((new_df, temp_df), axis=1)

        # this line is data at PM2:40
        new_df = pd.concat((new_df, df_230.shift(1-step_len)), axis=1)

        new_df = pd.concat((new_df, result.shift(0-step_len)), axis=1)
        new_df = new_df.iloc[:0-step_len]
        new_df.to_csv(file_train, columns=new_df.columns, mode='w', header=True, encoding="utf_8_sig")

    def gen_train_data_for_single(self, step_len):
        stocks = self.get_valid_single_stock_list()
        for stock in stocks:
            self.gen_train_data_for_single_by_stock(stock, step_len)

    # the last day's data, calculated from p230
    def gen_p230_for_stock_by_date_list(self, st_code, date_list):
        dir = os.path.join(self.stock_dir, st_code)
        if not os.path.exists(dir):
            return None

        data = []
        for date in date_list:
            file_path = os.path.join(dir, st_code + '_' + date + '.csv')
            open_p, high, low, close, vol, amount, p230 = self.get_p230_in_date_file(file_path)

            if p230 is None:
                return None

            data.append([date, open_p, high, low, close, vol, amount, p230])

        cols = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df_p230 = pd.DataFrame(data, columns=cols)
        p230_path = os.path.join(self.p230_dir, st_code + '.csv')
        df_p230.to_csv(p230_path, columns=cols, mode='a', header=False, encoding="utf_8_sig")
        return df_p230.reset_index(drop=True)

    def gen_single_data_for_stock_by_date_list(self, st_code, df, df_p230):
        df_new = pd.concat((df, df_p230), axis=1)
        new_cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        file_path = os.path.join(self.single_data_dir, 's_'+st_code+'.csv')
        df_new.to_csv(file_path, columns=new_cols, mode='a', header=False, encoding="utf_8_sig")
        return df_new[new_cols].reset_index(drop=True)

    def gen_train_data_for_single_delta(self, st_code, cnt):
        train_data_dir = self.single_data_dir + '_' + str(self.step_len)
        file = os.path.join(self.single_data_dir, 's_'+st_code + '.csv')
        file_train = os.path.join(train_data_dir, st_code+'.csv')
        cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')

        df = df[0-self.step_len-cnt:].reset_index(drop=True)
        #df = pd.concat((df, df_with_p230[cols]), axis=0)

        file_230 = os.path.join(self.p230_dir, st_code + '.csv')
        cols_p230 = ['open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df_p230 = pd.read_csv(file_230, header=0, usecols=cols_p230, encoding='utf-8')
        df_p230 = df_p230[-1-cnt:-1].reset_index(drop=True)
        df_p230.rename(columns={'open': 'open_l',
                                'high': 'high_l',
                                'low': 'low_l',
                                'close': 'close_l',
                                'vol': 'vol_l',
                                'amount': 'amount_l',
                                'p230': 'p230_l'}, inplace=True)

        result = df[['close']]
        result = result[0-cnt:].reset_index(drop=True)
        result.columns = ['result']

        new_df = df[:cnt]

        for i in range(1, self.step_len-1):
            temp_df = df.shift(0-i)[:cnt]
            temp_df.rename(columns={'open': 'open' + '_' + str(i),
                                    'high': 'high' + '_' + str(i),
                                    'low': 'low' + '_' + str(i),
                                    'close': 'close' + '_' + str(i),
                                    'vol': 'vol' + '_' + str(i),
                                    'amount': 'amount' + '_' + str(i),
                                    'p230': 'p230' + '_' + str(i)}, inplace=True)

            new_df = pd.concat((new_df, temp_df), axis=1)

        # this line is data at PM2:40
        new_df = pd.concat([new_df, df_p230], axis=1)
        new_df = pd.concat([new_df, result], axis=1)
        new_df.to_csv(file_train, columns=new_df.columns, mode='a', header=False, encoding="utf_8_sig")

    def update_data_for_stocks(self):
        stocks = self.get_valid_single_stock_list()
        for stock in stocks:
            st_code = stock.split('.')[0]
            # down stock data using tushare
            date_list, df_stock = self.downloadClient.get_data_for_stock(stock)

            if (date_list is None) or (len(date_list) == 0):
                continue

            # gen p230
            df_p230 = self.gen_p230_for_stock_by_date_list(st_code, date_list)
            if df_p230 is None:
                print("update_data_for_stocks() gen p230 for stock %s failed!" % st_code)
                break

            # gen single stock data
            df_with_p230 = self.gen_single_data_for_stock_by_date_list(st_code, df_stock, df_p230['p230'])

            # gen single train data
            self.gen_train_data_for_single_delta(st_code, len(date_list))

    def download_for_stocks_2(self, skip_date):
        stocks = self.get_all_stocks(2)
        for stock in stocks:
            # down stock data using tushare
            self.downloadClient.get_data_for_stock_no_fenbi(stock, skip_date)

    # the last day's data, calculated from p230
    def gen_today_p230_for_stock(self, st_code):
        date = datetime.datetime.now().strftime('%Y%m%d')

        data = []
        file_path = os.path.join(self.stock_today_dir, st_code + '_' + date + '.csv')
        open_p, high, low, close, vol, amount, p230 = self.get_p230_in_date_file(file_path)

        if p230 is None:
            return False

        data.append([date, open_p, high, low, close, vol, amount, p230])

        cols = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df_p230 = pd.DataFrame(data, columns=cols)
        p230_path = os.path.join(self.p230_dir, st_code + '.csv')
        df_p230.to_csv(p230_path, columns=cols, mode='a', header=False, encoding="utf_8_sig")
        return True

    def gen_today_p230(self):
        stocks = self.get_valid_single_stock_list()
        p230_stocks = []
        for stock in stocks:
            st_code = stock.split('.')[0]
            if self.downloadClient.get_today_fenbi_for_stock(st_code):
                p230_stocks.append(stock)

        p230_stocks_2 = []
        for stock in p230_stocks:
            if self.gen_p230_for_stock_by_date_list(stock):
                p230_stocks_2.append(stock)

        return p230_stocks_2

    def get_p230_redict_last_input(self, st_code):
        today = datetime.datetime.now().strftime('%Y%m%d')
        file = os.path.join(self.single_data_dir, 's_'+st_code + '.csv')
        cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')

        values = df[1-self.step_len:].reset_index(drop=True).values.astype('float32')

        file_230 = os.path.join(self.p230_dir, st_code + '.csv')
        cols_p230 = ['open', 'high', 'low', 'close', 'vol', 'amount', 'p230']
        df_p230 = pd.read_csv(file_230, header=0, usecols=cols_p230, encoding='utf-8')
        p230_values = df_p230[-1:].values.astype('float32')

        dataset = values[0]
        for i in range(1, values.shape[0]):
            dataset = np.concatenate([dataset, values[i]], axis=1)

        dataset = np.concatenate([dataset, p230_values], axis=1)

        return dataset


if __name__ == '__main__':
    dataUtil = DataUtil('../')
    # dataUtil.collect_data_for_stocks()
    # df = ts.get_tick_data('002001', date='20181203', src='tt')
    # df = ts.get_realtime_quotes('000581')
    # print(df)

    # 生成 PM2:40的数据
    #dataUtil.gen_p230_for_stocks()

    # generate valid stock_list for single predict
    # 检查对应的日期数据是否一致
    #dataUtil.gen_single_stock_list()

    # generate dataset for single predict
    #dataUtil.gen_single_data()

    #dataUtil.gen_train_data_for_single(10)

    #dataUtil.update_data_for_stocks()

    dataUtil.download_for_stocks_2('20190118')

    #dataUtil.gen_today_p230()
