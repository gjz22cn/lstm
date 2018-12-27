#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import tushare as ts


class DownloadClient:
    def __init__(self):
        self.dataDir = '..\data'
        ts.set_token('b1de6890364825a4b7b2d227b64c09a486239daf67451c5638404c62')
        self.pro = ts.pro_api()

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

    def getStockDailyInfo(self, code, name):
        df = self.pro.daily(ts_code=code, start_date='20160101', end_date='20181130')
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
        fileFullPath = os.path.join(self.dataDir, code + '.csv')
        mode = 'w'
        needHeader=True
        if os.path.exists(fileFullPath):
            mode = 'a'
            needHeader=False

        # colunms 保存指定的列索引
        columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']

        df = df.sort_values(by='trade_date', ascending=True)

        df.to_csv(fileFullPath, columns=columns, mode=mode, header=needHeader, encoding="utf_8_sig")


if __name__ == '__main__':
    downloadClient = DownloadClient()
    stockList = downloadClient.getStockList()
    count = 0
    for i in stockList.index:
        downloadClient.getStockDailyInfo(stockList.loc[i]['ts_code'], stockList.loc[i]['name'])
        count += 1
        if count == 5:
            break
