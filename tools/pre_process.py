import os
import numpy as np
import pandas as pd
import tushare as ts


# df = ts.get_tick_data('002001', date='2018-11-29', src='tt')
# print(df)

class DataUtil:
    def __init__(self, dir):
        self.root_dir = dir
        self.start_date = '20140101'
        self.end_date = '20181130'
        self.stock_dir = os.path.join(self.root_dir, 'stock')
        ts.set_token('b1de6890364825a4b7b2d227b64c09a486239daf67451c5638404c62')
        self.pro = ts.pro_api()

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
        file = os.path.join(self.root_dir, 'ori/stock_list.csv')
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


if __name__ == '__main__':
    #dataUtil = DataUtil('../')
    #dataUtil.collect_data_for_stocks()
    #df = ts.get_tick_data('002001', date='20181203', src='tt')
    df = ts.get_realtime_quotes('000581')
    print(df)



