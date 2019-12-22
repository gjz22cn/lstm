import os
import numpy as np
import pandas as pd
import tushare as ts
from DownloadClient import DownloadClient
import time
import datetime
import threading
from time import sleep, ctime
import matplotlib.pyplot as plt
import seaborn


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

    def stock_list_del_index(self, idx):
        file = os.path.join(self.root_dir, 'stock_list.csv')
        file_out = os.path.join(self.root_dir, 'stock_list_n.csv')
        if not os.path.exists(file):
            return []

        df = pd.read_csv(file, header=0, index_col=0, encoding='utf-8')
        df = df.drop(index=idx).reset_index()
        columns = ['ts_code',	'symbol', 'name', 'area', 'industry', 'fullname', 'enname',
                   'market', 'exchange', 'curr_type', 'list_status', 'list_date', 'delist_date', 'is_hs']
        df.to_csv(file_out, columns=columns, encoding='utf-8')

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

    def download_for_stocks_2(self, type, skip_date):
        stocks = self.get_all_stocks(type)
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

    def concat_in_one_stock_by_date(self, ts_code):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.stocks_dir, st_code + '.csv')
        cols = ['trade_date', 'close']
        df_stock = pd.read_csv(file, header=0, usecols=cols, index_col=0, encoding='utf-8')
        if df_stock is None:
            return True

        df_stock.rename(columns={'trade_date': 'trade_date', 'close': ts_code}, inplace=True)

        file2 = os.path.join(self.root_dir, 'stocks_by_date.csv')
        if not os.path.exists(file2):
            df_stock.to_csv(file2, columns=df_stock.columns, mode='w', header=True, encoding="utf_8_sig")
            return True

        df_curr = pd.read_csv(file2, header=0, index_col=0, encoding='utf-8')
        if df_curr is None:
            df_stock.to_csv(file2, columns=df_stock.columns, mode='w', header=True, encoding="utf_8_sig")
            return True

        if ts_code in df_curr.columns:
            return False

        #df_all = pd.concat([df_curr, df_stock], axis=1, join_axes=[df_curr.index])
        df_all = pd.merge(df_curr, df_stock, on="trade_date", how="left")
        df_all.to_csv(file2, columns=df_all.columns, mode='w', header=True, encoding="utf_8_sig")
        return True

    def concat_all_stocks_by_date(self):
        stocks = self.get_all_stocks(3)
        count = 0
        for stock in stocks:
            if self.concat_in_one_stock_by_date(stock):
                count = count + 1
                print (stock)

    def concat_in_one_stock_by_date_range(self, ts_code, start_date, end_date):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.stocks_dir, st_code + '.csv')
        cols = ['trade_date', 'close']
        df_stock = pd.read_csv(file, header=0, usecols=cols, index_col=0, encoding='utf-8')
        if df_stock is None:
            return True

        df_stock.rename(columns={'trade_date': 'trade_date', 'close': ts_code}, inplace=True)
        df_stock = df_stock.loc[start_date:end_date]

        file2 = os.path.join(self.root_dir, 'stocks_by_date_'+str(start_date) + '_' + str(end_date) + '.csv')
        if not os.path.exists(file2):
            df_stock.to_csv(file2, columns=df_stock.columns, mode='w', header=True, encoding="utf_8_sig")
            return True

        df_curr = pd.read_csv(file2, header=0, index_col=0, encoding='utf-8')
        if df_curr is None:
            df_stock.to_csv(file2, columns=df_stock.columns, mode='w', header=True, encoding="utf_8_sig")
            return True

        if ts_code in df_curr.columns:
            return False

        #df_all = pd.concat([df_curr, df_stock], axis=1, join_axes=[df_curr.index])
        df_all = pd.merge(df_curr, df_stock, on="trade_date", how="left")
        df_all.to_csv(file2, columns=df_all.columns, mode='w', header=True, encoding="utf_8_sig")
        return True

    def concat_all_stocks_by_date_range(self, start_date, end_date):
        stocks = self.get_all_stocks(3)
        count = 0
        for stock in stocks:
            if self.concat_in_one_stock_by_date_range(stock, start_date, end_date):
                count = count + 1
                print(stock)

    def concat_all_stocks_by_date_range_new(self, start_date, end_date):
        file_out = os.path.join(self.root_dir, 'stocks_by_date_' + str(start_date) + '_' + str(end_date) + '_2.csv')

        stocks = self.get_all_stocks(3)
        df_out = None
        for stock in stocks:
            st_code = stock.split('.')[0]
            file_stock = os.path.join(self.stocks_dir, st_code + '.csv')
            cols = ['trade_date', 'close']
            df_stock = pd.read_csv(file_stock, header=0, usecols=cols, index_col=0, encoding='utf-8')
            if df_stock is None:
                continue

            df_stock.rename(columns={'trade_date': 'trade_date', 'close': stock}, inplace=True)
            if start_date not in df_stock.index.values:
                continue

            if end_date not in df_stock.index.values:
                continue

            print(stock)
            df_stock = df_stock.loc[start_date:end_date]

            if df_out is None:
                df_out = df_stock
            else:
                df_out = pd.merge(df_out, df_stock, on="trade_date", how="left")

        df_out.to_csv(file_out, columns=df_out.columns, mode='w', header=True, encoding="utf_8_sig")


    def pull_up_one_stock_for_some_days(self, ts_code, days):
        file = os.path.join(self.root_dir, 'stocks_by_date.csv')
        df = pd.read_csv(file, header=0, index_col=0, encoding='utf-8')
        if ts_code not in df.columns:
            print("stock %s not found")%(ts_code)
            return

        df_stock = df[ts_code].reset_index()
        df_stock = df_stock.drop('trade_date', 1)
        df_stock = df_stock[days:].reset_index()
        df_new = pd.concat([df.drop(ts_code, 1).reset_index(), df_stock], axis=1)
        file_out = os.path.join(self.root_dir, 'stocks_by_date_'+ts_code+'_'+str(days)+'.csv')
        df_new.to_csv(file_out, columns=df_new.columns, mode='w', header=True, index=False, encoding="utf_8_sig")

    def stocks_corr(self, ts_code, days):
        file = os.path.join(self.root_dir, 'stocks_by_date_'+ts_code+'_'+str(days)+'.csv')
        file_corr = os.path.join(self.root_dir, 'stocks_corr_'+ts_code+'_'+str(days)+'.csv')
        df = pd.read_csv(file, header=0, index_col=0, encoding='utf-8').reset_index()
        corr_df = df.corr(method='pearson')
        corr_df.reset_index()
        corr_df.to_csv(file_corr, columns=[ts_code], mode='w', header=True, encoding="utf_8_sig")

    def calc_corr_for_stock_skip_days(self, ts_code, days):
        time_stamp = time.strftime("%Y%m%d", time.localtime())
        file_corr_all = os.path.join(self.root_dir, 'stocks_corr_all_' + time_stamp + '.csv')
        max_stock = [ts_code]
        columns = ['stock']

        for day in days:
            #dataUtil.pull_up_one_stock_for_some_days(ts_code, day)
            file_ori = os.path.join(self.root_dir, 'stocks_by_date.csv')
            df_ori = pd.read_csv(file_ori, header=0, index_col=0, encoding='utf-8')
            if ts_code not in df_ori.columns:
                print("stock %s not found") % (ts_code)
                columns.append(str(day))
                max_stock.append(ts_code)
                continue

            df_stock = df_ori[ts_code].reset_index()
            df_stock = df_stock.drop('trade_date', 1)
            df_stock = df_stock[day:].reset_index()
            df_ori2 = pd.concat([df_ori.drop(ts_code, 1).reset_index(), df_stock], axis=1).reset_index()
            df_ori2 = df_ori2.drop(columns=['level_0'])

            #dataUtil.stocks_corr(ts_code, day)
            df_corr = df_ori2.corr(method='pearson')
            df = df_corr[ts_code].reset_index()
            '''
            file = os.path.join(self.root_dir, 'stocks_corr_' + ts_code + '_' + str(day) + '.csv')
            df = pd.read_csv(file, header=0, index_col=0, encoding='utf-8')
            df.rename(columns={'trade_date': 'trade_date', ts_code: ts_code+'_'+str(day)}, inplace=True)
            '''
            df = df.set_index('index')
            df = df.drop(index=['index', ts_code])
            columns.append(str(day))
            max_stock.append(df.idxmax(axis=0).values[0])
            '''
            os.remove(file)
            file_date = os.path.join(self.root_dir, 'stocks_by_date_' + ts_code + '_' + str(day) + '.csv')
            os.remove(file_date)
            '''

        df_r = pd.DataFrame([max_stock], columns=columns)
        if os.path.exists(file_corr_all):
            df_r.to_csv(file_corr_all, mode='a', index=False, header=False, encoding="utf_8_sig")
        else:
            df_r.to_csv(file_corr_all, mode='a', index=False, header=True, encoding="utf_8_sig")

    def filter_stock_corr_values(self, ts_code):
        file = os.path.join(self.root_dir, 'stocks_corr_' + ts_code + '.csv')
        file2 = os.path.join(self.root_dir, 'stocks_corr_' + ts_code + '_skip.csv')
        df = pd.read_csv(file, header=0, index_col=0, encoding='utf-8')
        skip_index = []
        for index, row in df.iterrows():
            need_skip = True
            for item in row:
                if item < -0.93 or item > 0.93:
                    need_skip = False
                    break

            if need_skip:
                skip_index.append(index)

        df = df.drop(index=skip_index)
        df.to_csv(file2, columns=df.columns, mode='w', header=True, encoding="utf_8_sig")

    # calc most like stocks
    def calc_most_like_stocks_for_all(self, days, calc_n):
        time_stamp = time.strftime("%Y%m%d", time.localtime())
        file_corr_all = os.path.join(self.root_dir, 'stocks_corr_all_' + time_stamp + '.csv')
        file_corr_all2 = os.path.join(self.root_dir, 'stocks_corr_all_20190824.csv')

        df = None
        df2 = None
        if os.path.exists(file_corr_all):
            df = pd.read_csv(file_corr_all, header=0, index_col=0, encoding='utf-8')

        if os.path.exists(file_corr_all2):
            df2 = pd.read_csv(file_corr_all2, header=0, index_col=0, encoding='utf-8')

        count = 0
        stocks = self.get_all_stocks(3)
        for stock in stocks:
            if df is not None:
                if stock in df.index.values:
                    continue

            if df2 is not None:
                if stock in df2.index.values:
                    continue

            starttime = datetime.datetime.now()
            print("Calc stock: " + stock)
            self.calc_corr_for_stock_skip_days(stock, days)
            endtime = datetime.datetime.now()
            print("Calc stock: %s %ds" % (stock, (endtime - starttime).seconds))
            count = count + 1
            if count == calc_n:
                return

    # #############################################################################################
    # calc most like stocks for range start
    # #############################################################################################
    def calc_corr_for_stock_skip_days_range(self, ts_code, days, file_out):
        max_stock = [ts_code]
        columns = ['stock']

        for day in days:
            file_ori = os.path.join(self.root_dir, 'stocks_by_date.csv')
            df_ori = pd.read_csv(file_ori, header=0, index_col=0, encoding='utf-8')
            if ts_code not in df_ori.columns:
                print("stock %s not found") % (ts_code)
                columns.append(str(day))
                max_stock.append(ts_code)
                continue

            df_stock = df_ori[ts_code].reset_index()
            df_stock = df_stock.drop('trade_date', 1)
            df_stock = df_stock[day:].reset_index()
            df_ori2 = pd.concat([df_ori.drop(ts_code, 1).reset_index(), df_stock], axis=1).reset_index()
            df_ori2 = df_ori2.drop(columns=['level_0'])

            df_corr = df_ori2.corr(method='pearson')
            df = df_corr[ts_code].reset_index()

            df = df.set_index('index')
            df = df.drop(index=['index', ts_code])
            columns.append(str(day))
            max_stock.append(df.idxmax(axis=0).values[0])

        df_r = pd.DataFrame([max_stock], columns=columns)
        if os.path.exists(file_out):
            df_r.to_csv(file_out, mode='a', index=False, header=False, encoding="utf_8_sig")
        else:
            df_r.to_csv(file_out, mode='a', index=False, header=True, encoding="utf_8_sig")

    def calc_most_like_stocks_for_range(self, days, start_idx, end_idx):
        time_stamp = time.strftime("%Y%m%d", time.localtime())
        idx_range_str = '_' + str(start_idx) + '_' + str(end_idx)
        file_corr_range = os.path.join(self.root_dir, 'stocks_corr_all_' + time_stamp + idx_range_str + '.csv')

        df = None
        if os.path.exists(file_corr_range):
            df = pd.read_csv(file_corr_range, header=0, index_col=0, encoding='utf-8')

        count = 0
        stocks = self.get_all_stocks(3)
        for stock in stocks:
            count = count + 1

            if count < start_idx:
                continue

            if count > end_idx:
                return

            if df is not None:
                if stock in df.index.values:
                    continue

            start_time = datetime.datetime.now()
            print("Calc stock: " + stock)
            self.calc_corr_for_stock_skip_days_range(stock, days, file_corr_range)
            end_time = datetime.datetime.now()
            print("Calc stock: %s %ds" % (stock, (end_time - start_time).seconds))

    def calc_most_like_stocks_for_range_parallel(self, days, start_idx, steps):
        print("Starting at:", ctime())
        threads = []
        loops = range(10)

        for i in loops:
            idx_s = start_idx + i*steps
            idx_e = idx_s + steps
            t = threading.Thread(target=self.calc_most_like_stocks_for_range, args=(days, idx_s, idx_e))

        threads.append(t)
        for i in loops:
            threads[i].start()

        for i in loops:
            threads[i].join()

        print("All done at:", ctime())

    def concat_two_files(self, file_ins, out):
        df = None
        for file_in in file_ins:
            file = os.path.join(self.root_dir, file_in)
            df1 = pd.read_csv(file, encoding='utf-8')
            if df is None:
                df = df1
            else:
                df = pd.concat([df, df1])

        df = df.reset_index(drop=True)
        file_out = os.path.join(self.root_dir, out)
        df.to_csv(file_out, mode='w', index=False, encoding="utf_8_sig")
    # #############################################################################################
    # calc most like stocks for range start
    # #############################################################################################

    # calc chg for all stocks
    def calc_all_stocks_chg(self, days, delta):
        file_suf = ''
        base_index = []
        new_index = []
        for day in days:
            base_index.insert(0, -1-day)
            new_index.insert(0, -1-day+delta)
            file_suf = file_suf + '_' + str(day)

        file_ori = os.path.join(self.root_dir, 'stocks_by_date.csv')
        df_ori = pd.read_csv(file_ori, header=0, index_col=0, encoding='utf-8')
        df_base = df_ori.iloc[base_index, :].reset_index().drop(columns=['trade_date'])
        df_new = df_ori.iloc[new_index, :].reset_index().drop(columns=['trade_date'])
        df_delta = df_new - df_base
        df_chg = df_delta.div(df_base, fill_value=0)

        time_stamp = time.strftime("%Y%m%d", time.localtime())
        file_chg = os.path.join(self.root_dir, 'stocks_chg' + file_suf + '_' + time_stamp + '.csv')
        df_chg.to_csv(file_chg, mode='a', header=True, encoding="utf_8_sig")


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

    # download stock data to specific date
    #dataUtil.download_for_stocks_2(3, '20190831')

    # Step2: concat all stocks by date close price
    # dataUtil.concat_all_stocks_by_date()
    #dataUtil.concat_all_stocks_by_date_range_new(20170103, 20190830)

    # pull up one stock for specific days
    #dataUtil.pull_up_one_stock_for_some_days('000848.SZ', 5)

    # calc stocks corr
    #dataUtil.stocks_corr('000848.SZ')

    #dataUtil.calc_corr_for_stock_skip_days('000848.SZ', [5, 10, 15, 20])
    #dataUtil.filter_stock_corr_values('000848.SZ')

    #dataUtil.gen_today_p230()

    # Step3: calc chg for all stocks
    # dataUtil.calc_all_stocks_chg([5, 10, 15, 20], 5)

    # Step4: calc most like stocks
    # dataUtil.calc_most_like_stocks_for_all([5, 10, 15, 20], 200)
    #dataUtil.calc_most_like_stocks_for_range([5, 10, 15, 20], 400, 500)
    #dataUtil.calc_most_like_stocks_for_range_parallel([5, 10, 15, 20], 400, 200)

    dataUtil.stock_list_del_index(2682)
