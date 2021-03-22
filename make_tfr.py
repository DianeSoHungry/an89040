import os
import random
from glob import glob
from tqdm import tqdm
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import imp

imp.reload(utils)
FIELDS = ['time', 'open', 'high', 'low', 'close', 'avg', 'volume', 'ret_close0--1', 'ret_open6-1']
DAYS = 30

TFR_PATH = utils.get_cfg('TFR', 'DATA_PATH')
if not os.path.exists(os.path.dirname(TFR_PATH)):
    os.makedirs(os.path.dirname(TFR_PATH))


def dtstr_to_halfyear(dtstr):
    year = dtstr[:4]
    mon = dtstr[4:6]
    if mon <= '06':
        mon = '06'
    else:
        mon = '12'
    return year + mon


def suffle_keys(keylist, times=3):
    for i in range(times):
        random.shuffle(keylist)


def fill_pool():
    global POOL
    source_data_paths = glob(utils.get_cfg('SOURCE', 'DATA_PATH').replace('{}', '??????'))
    print('Filling Pool...')
    for p in tqdm(source_data_paths):
        df = pd.read_feather(p, columns=FIELDS)
        code = os.path.basename(p).split('.')[0]

        for r in range(DAYS, df.shape[0] + 1):
            date, arr, target = df.iloc[r - 1, 0], df.iloc[r - DAYS: r, 1:-1].values, df.iloc[r - 1, -1]
            dtstr = date.strftime('%Y%m%d')
            #     dtstr, arr.shape, target
            #     ('20140327', (30, 7), -0.0821917808219178)

            if np.isnan(arr).any() or np.isnan(target).any():
                continue

            keyname = '%s.%s' % (dtstr, code)
            POOL[keyname] = {'arr': arr, 'target': target}


def get_save_path(dtstr):
    save_path = TFR_PATH.format(YYYYmm=dtstr_to_halfyear(dtstr))
    return save_path


def save_tfr():
    print('Saving tfr...')
    global POOL
    pool_keylist = list(POOL.keys())
    suffle_keys(pool_keylist, times=3)

    path2handler = {}
    for pool_key in tqdm(pool_keylist):
        dtstr, code = pool_key.split('.')
        tfr_path = get_save_path(dtstr)
        if tfr_path not in path2handler:
            path2handler[tfr_path] = tf.io.TFRecordWriter(tfr_path)

    for pool_key in tqdm(pool_keylist):
        dtstr, code = pool_key.split('.')
        arr = POOL[pool_key]['arr']
        target = POOL[pool_key]['target']
        tfr_path = get_save_path(dtstr)

        feature = {
            'arr': tf.train.Feature(float_list=tf.train.FloatList(value=arr.flatten())),
            'target': tf.train.Feature(float_list=tf.train.FloatList(value=[target])),
            'dtstr': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dtstr.encode()])),
            'code': tf.train.Feature(bytes_list=tf.train.BytesList(value=[code.encode()]))

        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        path2handler[tfr_path].write(example.SerializeToString())

    for k, v in path2handler.items():
        v.close()


# DATA_PATH = utils.get_cfg('PICS', 'DATA_PATH')
POOL = {}


def pipeline():
    fill_pool()
    save_tfr()


if __name__ == '__main__':
    pipeline()
