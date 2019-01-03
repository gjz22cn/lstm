import os
import numpy as np
from pandas import read_csv,DataFrame
from sklearn.preprocessing import MinMaxScaler

dst_dir = "../data_p"
g_preLen = 30

def prepare_dataset(data, time_steps):
    cnt = data.shape[0] - time_steps
    data_x = data[:cnt]
    for i in range(1, time_steps):
        data_x = np.concatenate([data_x, data[i:i + cnt]], axis=1)

    np.concatenate([data_x, data[time_steps:, 3:4]], axis=1)
    return data_x

def convert_to_data_set(path, file_name):
    cols = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    data = read_csv(path, header=0, usecols=cols, encoding='utf-8')
    values = data.values.astype('float32')
    if values.shape[0] < g_preLen+1:
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(values)
    dataset = prepare_dataset(dataset, g_preLen)

    df = DataFrame(dataset)
    df.to_csv(os.path.join(dst_dir, file_name))

def join_files(dir):
    # 列出文件夹下所有的目录与文件
    list = os.listdir(dir)

    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isfile(path):
            convert_to_data_set(path, list[i])

join_files("../data")