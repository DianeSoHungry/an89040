import os
import sys
from tqdm import tqdm
import utils

sys.path.insert(0, '/home/dev_qy/nhy/api')
import dailytables2 as tb2

DATA_PATH = utils.get_cfg('SOURCE', 'DATA_PATH')

DATA_DIR = os.path.dirname(DATA_PATH)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

import dailytables

RET_TABLE = dailytables.get_5dret_open()


def standerize(df):
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    return df.sub(df_mean, axis=0).div(df_std, axis=0)


RET_TABLE = standerize(RET_TABLE)


def save():
    print('Reading...')
    grp = tb2.read_1d_table_grp_by_stock(fields=['open', 'high', 'low', 'close', 'avg', 'volume'],
                                         start='2008-',
                                         end='2020-')
    print('Saving...')
    for code, g in tqdm(grp):
        g = g.set_index('time').drop(columns=['code'])
        g['ret_close0--1'] = g['close'] / g['close'].shift(1) - 1
        if code not in RET_TABLE.columns: \
                continue
        g['ret_open6-1'] = RET_TABLE[code].shift(-1)
        #         g['ret_open11-1'] = g['open'].shift(-11) / g['open'].shift(-1) - 1
        g.reset_index().to_feather(DATA_PATH.format(code))


if __name__ == '__main__':
    save()
