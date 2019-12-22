import os
import numpy as np
import pandas as pd
from StUtil import StUtil
import matplotlib.pyplot as plt
import time
import datetime
import threading
from time import sleep, ctime


class ZbUtil:
    def __init__(self, dir):
        self.root_dir = dir
        self.start_date = '20140101'
        self.end_date = '20181130'
        self.stocks_dir = os.path.join(self.root_dir, 'stocks')
        self.stock_dir = os.path.join(self.root_dir, 'stock')
        self.stock_today_dir = os.path.join(self.root_dir, 'stock')
        self.p230_dir = os.path.join(self.root_dir, 'p230')
        self.single_data_dir = os.path.join(self.root_dir, 'single_d')
        self.corr_dir = os.path.join(self.root_dir, 'corr')
        self.zb_dir = os.path.join(self.root_dir, 'zb')
        self.stUtil = StUtil(self.root_dir)

    def calc_stock_kdj(self, ts_code):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.stocks_dir, st_code + '.csv')

        cols = ['trade_date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount']
        df = pd.read_csv(file, header=0, usecols=cols, dtype={'trade_date': str}, encoding='utf-8')

        if df is None:
            return None

        df = df[-200:]
        try:
            low_list = df['low'].rolling(9, min_periods=9).min()
            low_list.fillna(value=df['low'].expanding().min(), inplace=True)
            high_list = df['high'].rolling(9, min_periods=9).max()
            high_list.fillna(value=df['high'].expanding().max(), inplace=True)
            rsv = (df['close'] - low_list) / (high_list - low_list) * 100

            df['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
            df['D'] = df['K'].ewm(com=2).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
        except:
            print("calc kdj for %s failed!"%(ts_code))
            return None

        df = pd.concat((df, pd.DataFrame(columns=['KDJ_label', 'stock'])), axis=1)
        kdj_position = df['K'] > df['D']
        df.loc[kdj_position[(kdj_position == True) & (kdj_position.shift() == False)].index - 1, 'KDJ_label'] = '金叉'
        df.loc[kdj_position[(kdj_position == False) & (kdj_position.shift() == True)].index - 1, 'KDJ_label'] = '死叉'

        df_tail = df.iloc[-1:].tail(1).reset_index(drop=True)
        df_pre_tail = df.iloc[-2:-1].reset_index(drop=True)

        df_tail = df_tail.loc[:, ['trade_date', 'stock', 'K', 'D', 'J', 'KDJ_label']]
        df_tail.loc[0, 'stock'] = ts_code

        if df_tail.loc[0, 'K'] < 20 and df_tail.loc[0, 'D'] < 20 and df_tail.loc[0, 'J'] < 20:
            return df_tail
        elif df_tail.loc[0, 'KDJ_label'] == '金叉' or df_pre_tail.loc[0, 'KDJ_label'] == '金叉':
            if df_tail.loc[0, 'K'] < 30 and df_tail.loc[0, 'D'] < 30 and df_tail.loc[0, 'J'] < 30:
                df_tail.loc[0, 'KDJ_label'] = '金叉30'
            elif df_tail.loc[0, 'K'] < 40 and df_tail.loc[0, 'D'] < 40 and df_tail.loc[0, 'J'] < 40:
                df_tail.loc[0, 'KDJ_label'] = '金叉40'
            elif df_tail.loc[0, 'K'] < 50 and df_tail.loc[0, 'D'] < 50 and df_tail.loc[0, 'J'] < 50:
                df_tail.loc[0, 'KDJ_label'] = '金叉50'
            else:
                df_tail.loc[0, 'KDJ_label'] = '金叉50+'
            return df_tail
        else:
            return None

        '''
        print(df.loc[:, ['K', 'D', 'J']].head(20))
        print(df.loc[:, ['K', 'D', 'J']].tail(10))

        df['KDJ_金叉死叉'] = ''
        kdj_position = df['K'] > df['D']
        df.loc[kdj_position[(kdj_position == True) & (kdj_position.shift() == False)].index-1, 'KDJ_金叉死叉'] = '金叉'
        df.loc[kdj_position[(kdj_position == False) & (kdj_position.shift() == True)].index-1, 'KDJ_金叉死叉'] = '死叉'

        file_out = os.path.join(self.zb_dir, st_code+'.csv')
        df.to_csv(file_out, mode='w', index=False, encoding="utf_8_sig")
        '''

    def kdj_filter(self, type):
        df_out = None
        stocks = self.stUtil.get_all_stocks(type)
        for stock in stocks:
            df_stock = self.calc_stock_kdj(stock)
            if df_out is None:
                df_out = df_stock
            else:
                df_out = pd.concat((df_out, df_stock), axis=0)

        print(df_out)
        file_out = os.path.join(self.zb_dir, 'kdj_20191221.csv')
        df_out.to_csv(file_out, mode='w', index=False, encoding="utf_8_sig")


if __name__ == '__main__':
    zbUtil = ZbUtil('../')
    zbUtil.kdj_filter(3)
