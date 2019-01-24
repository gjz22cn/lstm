#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import time
import tushare as ts
import pandas as pd
import datetime


class DownloadClient:
    def __init__(self):
        self.root_dir = '../'
        self.stocks_dir = os.path.join(self.root_dir, 'stocks')
        self.zhishus_dir = os.path.join(self.root_dir, 'zhishus')
        self.dataDir = '..\stocks'
        ts.set_token('b1de6890364825a4b7b2d227b64c09a486239daf67451c5638404c62')
        self.pro = ts.pro_api()
        self.start_date = '20140101'
        self.end_date = '20181231'
        self.sleep_cnt = 0
        self.zhishu_list = ['000001.SH', '000016.SH', '000300.SH',
                            '000905.SH', '399001.SZ', '399005.SZ',
                            '399006.SZ', '399016.SZ']

    # 查询当前所有正常上市交易的股票列表
    def getStockList(self):
        '''
        名称        类型    描述
        ts_code	    str	    TS代码
        symbol      str	    股票代码
        name	    str	    股票名称
        area	    str	    所在地域
        industry	str	    所属行业
        fullname	str	    股票全称
        enname	    str	    英文全称
        market	    str	    市场类型 （主板/中小板/创业板）
        exchange	str	    交易所代码
        curr_type	str	    交易货币
        list_status	str	    上市状态： L上市 D退市 P暂停上市
        list_date	str	    上市日期
        delist_date	str	    退市日期
        is_hs	    str	    是否沪深港通标的，N否 H沪股通 S深股通
        '''
        # colunms 保存指定的列索引
        stockList = self.pro.stock_basic(exchange='', list_status='L',
                                         fields='ts_code,symbol,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs')
        stockList.to_csv(os.path.join(self.dataDir, 'stock_list.csv'),
                         columns=['ts_code', 'symbol', 'name', 'area', 'industry', 'fullname', 'enname', 'market',
                                  'exchange',
                                  'curr_type', 'list_status', 'list_date', 'delist_date', 'is_hs'],
                         encoding="utf_8_sig")
        return stockList

    def getStockDailyInfo(self, code):
        today = datetime.datetime.now().strftime('%Y%m%d')
        df = self.pro.daily(ts_code=code, start_date='20', end_date=today)
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

        fileFullPath = os.path.join(self.dataDir, code.split('.')[0] + '.csv')
        mode = 'w'
        needHeader=True
        #if os.path.exists(fileFullPath):
        #    mode = 'a'
        #    needHeader=False

        # colunms 保存指定的列索引
        columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']

        df = df.sort_values(by='trade_date', ascending=True)

        df.to_csv(fileFullPath, columns=columns, mode=mode, header=needHeader, encoding="utf_8_sig")

    def get_fenbi_for_stock(self, st_code):
        stock_file = os.path.join('../stocks', st_code+'.csv')
        if not os.path.exists(stock_file):
            return

        dst_dir = os.path.join('../stock', st_code)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        columns = ['time', 'price', 'change', 'volume', 'amount', 'type']
        df_s = pd.read_csv(stock_file, header=0, usecols=['trade_date'], dtype={'trade_date': str}, encoding='utf-8')
        for index, row in df_s.iterrows():
            date = row['trade_date']

            if not date.startswith('201812'):
                continue

            file_path = os.path.join(dst_dir, st_code + '_' + date + '.csv')
            if os.path.exists(file_path):
                print("skip %s %s" % (st_code, date))
                continue

            self.sleep_cnt += 1
            if self.sleep_cnt % 5 == 0:
                time.sleep(1)

            df = ts.get_tick_data(st_code, date=date, src='tt')
            if df is None:
                print("%s %s is None" % (st_code, date))
                continue

            df = df.sort_values(by='time', ascending=True)
            df.to_csv(file_path, columns=columns, mode='w', header=True, encoding="utf_8_sig")

    def get_fenbi_for_stocks(self):
        dir = '../stocks'
        list = os.listdir(dir)
        for file in list:
            if not file.endswith('.csv'):
                continue

            if file.startswith('002') or file.startswith('300'):
                self.get_fenbi_for_stock(file[:-4])

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

    def get_fenbi_for_stock_by_date_list(self, st_code, date_list):
        columns = ['time', 'price', 'change', 'volume', 'amount', 'type']
        count = 0

        dst_dir = os.path.join('../stock', st_code)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for date in date_list:
            file_path = os.path.join(dst_dir, st_code + '_' + date + '.csv')
            if os.path.exists(file_path):
                print("skip %s %s" % (st_code, date))
                continue

            count += 1
            if count % 5 == 0:
                time.sleep(1)

            df = ts.get_tick_data(st_code, date=date, src='tt')
            if df is None:
                print("%s %s is None" % (st_code, date))
                continue

            df = df.sort_values(by='time', ascending=True)
            df.to_csv(file_path, columns=columns, mode='w', header=True, encoding="utf_8_sig")

    def get_data_for_stock(self, ts_code, skip_date):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.stocks_dir, st_code+'.csv')
        cols = ['trade_date']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')
        last_date = df[-1:]['trade_date'].values[0]

        if last_date == skip_date:
            return

        last_datetime = datetime.datetime.strptime(str(last_date), "%Y%m%d")
        delta = datetime.timedelta(days=1)
        start_datetime = last_datetime + delta
        start_date = start_datetime.strftime('%Y%m%d')
        today = datetime.datetime.now().strftime('%Y%m%d')
        df_new = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=today)
        columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol',
                   'amount']
        df_new = df_new.sort_values(by='trade_date', ascending=True)
        df_new.to_csv(file, columns=columns, mode='a', header=False, encoding="utf_8_sig")
        date_list = df_new['trade_date'].values.flatten()
        self.get_fenbi_for_stock_by_date_list(st_code, date_list)
        return date_list, df_new.reset_index(drop=True)

    def get_data_for_stock_no_fenbi(self, ts_code, skip_date):
        st_code = ts_code.split('.')[0]
        file = os.path.join(self.stocks_dir, st_code+'.csv')
        cols = ['trade_date']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')
        if df is None:
            return

        if df.shape[0] == 0:
            return

        last_date = df[-1:]['trade_date'].values[0]
        if str(last_date) == skip_date:
            print("skip", ts_code)
            return

        self.sleep_cnt += 1
        if self.sleep_cnt%5 == 0:
            time.sleep(1)

        last_datetime = datetime.datetime.strptime(str(last_date), "%Y%m%d")
        delta = datetime.timedelta(days=1)
        start_datetime = last_datetime + delta
        start_date = start_datetime.strftime('%Y%m%d')
        today = datetime.datetime.now().strftime('%Y%m%d')
        df_new = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=today)
        columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol',
                   'amount']
        df_new = df_new.sort_values(by='trade_date', ascending=True)
        df_new.to_csv(file, columns=columns, mode='a', header=False, encoding="utf_8_sig")

    def get_current_all(self):
        df = ts.get_today_all()
        print(df)

    def get_today_fenbi_for_stock(self, st_code):
        dst_dir = '../stock_today'
        columns = ['time', 'price', 'change', 'volume', 'amount', 'type']
        today = datetime.datetime.now().strftime('%Y%m%d')
        file_path = os.path.join(dst_dir, st_code + '_' + today + '.csv')

        df = ts.get_today_ticks(st_code)
        if df is None:
            print("%s %s is None" % (st_code, today))
            return False

        df = df.sort_values(by='time', ascending=True)
        df.to_csv(file_path, columns=columns, mode='w', header=True, encoding="utf_8_sig")
        return True

    def get_zhishu(self, ts_code):
        today = datetime.datetime.now().strftime('%Y%m%d')
        df = self.pro.index_daily(ts_code=ts_code, start_date='20140101', end_date=today)
        df = df.sort_values(by='trade_date', ascending=True)
        file = os.path.join(self.zhishus_dir, ts_code+'.csv')
        df.to_csv(file, mode='w', header=True, encoding="utf_8_sig")

    def get_zhishus(self):
        for zhishu in self.zhishu_list:
            self.get_zhishu(zhishu)

    def get_start_date(self, file):
        cols = ['trade_date']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')
        if df is None:
            return None

        last_date = df[-1:]['trade_date'].values[0]
        last_datetime = datetime.datetime.strptime(str(last_date), "%Y%m%d")
        delta = datetime.timedelta(days=1)
        start_datetime = last_datetime + delta
        start_date = start_datetime.strftime('%Y%m%d')
        return start_date

    def update_zhishu(self, ts_code):
        file = os.path.join(self.zhishus_dir, ts_code + '.csv')
        start_date = self.get_start_date(file)
        need_header = False
        mode = 'a'
        if start_date is None:
            start_date = '20140101'
            need_header = False
            mode = 'w'

        today = datetime.datetime.now().strftime('%Y%m%d')
        if start_date > today:
            return

        df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=today)
        df = df.sort_values(by='trade_date', ascending=True)
        df.to_csv(file, mode=mode, header=need_header, encoding="utf_8_sig")

    def update_all_zhishu(self):
        for zhishu in self.zhishu_list:
            self.update_zhishu(zhishu)

'''
if __name__ == '__main__':
    downloadClient = DownloadClient()
    downloadClient.update_data_for_stocks()
    
    stockList = downloadClient.getStockList()

    count = 0
    for i in stockList.index:
        if not stockList.loc[i]['ts_code'].startswith('6003'):
            continue
        downloadClient.getStockDailyInfo(stockList.loc[i]['ts_code'])
        if count%5 == 0:
            time.sleep(1)

        count += 1

    print("count=", count)
    
    #downloadClient.get_fenbi_for_stocks()
'''