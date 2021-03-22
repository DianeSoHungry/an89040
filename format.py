import os
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import sys

import tensorflow as tf
from tensorflow.keras.layers import Layer, Flatten, Dense, ReLU, Dropout, BatchNormalization, AvgPool1D, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import utils

FACTOR_PATH = utils.get_cfg('FACTOR', 'DATA_PATH')
factor_paths = sorted(glob(FACTOR_PATH.replace('{YYYYmm}', '??????')))


def read_factor(paths, signal_delay):
    '''在date天的singal，signal_delay天后发出
    '''
    with Parallel(n_jobs=6) as parallel:
        bt = pd.concat(parallel(delayed(pd.read_feather)(p, columns=['date', 'ticker', 'score']) for p in tqdm(paths)),
                       sort=False)
    #     bt.sort_values(by=['date', 'ticker'], inplace=True)
    bt.reset_index(drop=True, inplace=True)
    bt = bt.pivot_table(values='score', index='date', columns='ticker').shift(signal_delay)
    bt = bt.unstack().reset_index().rename(columns={0: 'score'})
    bt = bt[bt['score'].notna()]
    bt['date'] = pd.to_datetime(bt['date'])
    bt.reset_index(drop=True, inplace=True)
    #     bt = bt.set_index(['date', 'ticker'])
    return bt


if __name__ == '__main__':
    name = sys.argv[1]
    df = read_factor(factor_paths, 1)

    save_path = '/home/newVolume/factors/an/{}.fea'.format(name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_feather(save_path)
