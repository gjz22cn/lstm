import os
import numpy as np
import pandas as pd
import tushare as ts
from DownloadClient import DownloadClient
import math
import time
import datetime
import threading
from time import sleep, ctime


class StCorr:
    def __init__(self, dir):
        self.root_dir = dir
        self.corr_dir = os.path.join(self.root_dir, 'corr')
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
        self.eva_date = ''

    def set_eva_date(self, eva_date):
        self.eva_date = eva_date

    def fill_chg_to_corr(self):
        file_corr = os.path.join(self.corr_dir, 'stocks_corr_all_' + self.eva_date + '.csv')
        file_chg = os.path.join(self.corr_dir, 'stocks_chg_' + self.eva_date + '.csv')
        file_corr_chg = os.path.join(self.corr_dir, 'stocks_corr_chg_' + self.eva_date + '.csv')

        df_corr = pd.read_csv(file_corr, encoding='utf-8')
        df_chg = pd.read_csv(file_chg, encoding='utf-8')

        array_corr_chg = []
        for index, row in df_corr.iterrows():
            stock = row['stock']
            if stock not in df_chg.columns:
                continue
            chg_5 = df_chg.loc[3, stock]
            chg_10 = df_chg.loc[2, stock]
            chg_15 = df_chg.loc[1, stock]
            chg_20 = df_chg.loc[0, stock]
            chg_total = chg_5 + chg_10 + chg_15 + chg_20
            if chg_5 <= 0 or chg_10 <= 0 or chg_15 <= 0 or chg_20 <= 0:
                continue

            if math.isnan(chg_5) or math.isnan(chg_10) or math.isnan(chg_15) or math.isnan(chg_20):
                continue

            array_corr_chg.append([stock, chg_5, chg_10, chg_15, chg_20, chg_total])

        df_corr_chg = pd.DataFrame(array_corr_chg, columns=['stock', '5', '10', '15', '20', 'total'])
        df_corr_chg.to_csv(file_corr_chg, mode='w', index=False, header=True, encoding="utf_8_sig")


if __name__ == '__main__':
    stCorr = StCorr('../')
    stCorr.set_eva_date('20190830')
    stCorr.fill_chg_to_corr()
