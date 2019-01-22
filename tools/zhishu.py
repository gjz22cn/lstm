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

    def download_zhishu_all(self):
        self.downloadClient.get_zhishus()


if __name__ == '__main__':
    zhiShu = ZhiShu()
    zhiShu.download_zhishu_all()