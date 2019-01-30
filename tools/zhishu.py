import os
import numpy as np
import pandas as pd
import tushare as ts
from DownloadClient import DownloadClient
import datetime


class ZhiShu:
    def __init__(self):
        self.root_dir = '../'
        self.zhishus_dir = os.path.join(self.root_dir, 'zhishus')
        self.downloadClient = DownloadClient()
        self.zhishu_list = ['000001.SH', '000016.SH', '000300.SH',
                            '000905.SH', '399001.SZ', '399005.SZ',
                            '399006.SZ', '399016.SZ']

    def download_zhishu_all(self):
        self.downloadClient.get_zhishus()

    def update_zhishu_all(self):
        self.downloadClient.update_all_zhishu()

    def get_date_list(self, ts_code):
        file = os.path.join(self.zhishus_dir, ts_code + '.csv')
        cols = ['trade_date']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')
        if df is None:
            return []

        return df.values.flatten()

    @staticmethod
    def compare_date_list(d1, d2):
        if len(d1) != len(d2):
            return False

        for i in range(len(d1)):
            if d1[i] != d1[i]:
                return False

        return True

    def check_date_list(self):
        date_list_ref = None
        for zhishu in self.zhishu_list:
            if date_list_ref is None:
                date_list_ref = self.get_date_list(zhishu)
            else:
                date_list = self.get_date_list(zhishu)
                if not self.compare_date_list(date_list_ref, date_list):
                    return False

        return True

    def get_df(self, ts_code, with_trade_date):
        file = os.path.join(self.zhishus_dir, ts_code + '.csv')
        cols = ['close', 'vol', 'amount']
        if with_trade_date:
            cols = ['trade_date', 'close', 'vol', 'amount']
        df = pd.read_csv(file, header=0, usecols=cols, encoding='utf-8')
        return df

    def concat_all_zhishu(self):
        if not self.check_date_list():
            return

        df = self.get_df(self.zhishu_list[0], True)
        if df is None:
            return

        df.rename(columns={'close': self.zhishu_list[0] + '_close',
                           'vol': self.zhishu_list[0] + '_vol',
                           'amount': self.zhishu_list[0] + '_amount'}, inplace=True)

        for i in range(1, len(self.zhishu_list)):
            df_i = self.get_df(self.zhishu_list[i], False)
            if df_i is None:
                print("%s is broken!" % self.zhishu_list[i])
                return

            df_i.rename(columns={'close': self.zhishu_list[i] + '_close',
                                 'vol': self.zhishu_list[i] + '_vol',
                                 'amount': self.zhishu_list[i] + '_amount'}, inplace=True)

            df = pd.concat((df, df_i), axis=1)

        file = os.path.join(self.zhishus_dir, "all_zhishu.csv")
        df.to_csv(file, mode='w', header=True, float_format='%.3f', encoding="utf_8_sig")


if __name__ == '__main__':
    zhiShu = ZhiShu()
    #zhiShu.download_zhishu_all()
    zhiShu.update_zhishu_all()
    zhiShu.concat_all_zhishu()