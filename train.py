import os
from glob import glob
import pandas as pd
import numpy as np
# from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from models import Alphanet
import matplotlib.pyplot as plt
import tensorflow as tf
import utils

FEATURE_DESCRIPTION = {
    'arr': tf.io.FixedLenFeature([210], tf.float32),
    'target': tf.io.FixedLenFeature([1], tf.float32),
    'dtstr': tf.io.FixedLenFeature([], tf.string),
    'code': tf.io.FixedLenFeature([], tf.string)
}


def parse_example_code_n_date(example_string):
    feature_dict = tf.io.parse_example(example_string, FEATURE_DESCRIPTION)
    return feature_dict['dtstr'], feature_dict['code']


def parse_example(example_string):
    feature_dict = tf.io.parse_example(example_string, FEATURE_DESCRIPTION)
    feature_dict['arr'] = tf.reshape(feature_dict['arr'], (30, 7))
    feature_dict['arr'] = tf.transpose(feature_dict['arr'])
    feature_dict['target'] = feature_dict['target'] * 10
    return feature_dict['arr'], feature_dict['target']


def get_trainval_data_path_pairs():
    TFR_PATH = utils.get_cfg('TFR', 'DATA_PATH')
    paths = sorted(glob(TFR_PATH.replace('{YYYYmm}', '??????')))
    return zip(paths, paths[1:], paths[2:])


def save_history(history, figname):
    loss_path = utils.get_cfg('HISTORY', 'LOSS_PATH')
    if not os.path.exists(os.path.dirname(loss_path)):
        os.makedirs(os.path.dirname(loss_path))
    ax = plt.subplot(label='history' + figname)
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.legend()
    plt.title(figname)
    plt.show()
    plt.savefig(loss_path.format(YYYYmm=figname))


def save_outputstats(df, figname):
    path = utils.get_cfg('OUTPUT', 'OUTPUTSTATS_PATH')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    ax = plt.subplot(label='outputstats' + figname)
    ax.set_ylabel('score stats', color='r')
    ax.plot(df.groupby('date').apply(np.max)['score'], label='max')
    ax.plot(df.groupby('date').apply(np.mean)['score'], label='mean')
    ax.plot(df.groupby('date').apply(np.min)['score'], label='min')
    ax1 = ax.twinx()
    ax1.set_ylabel('stock#', color='b')
    ax1.plot(df.groupby('date')['score'].count(), label='stock#', c='b')
    ax.legend()
    plt.xticks(rotation=60)
    plt.title(figname)
    plt.show()
    plt.savefig(path.format(YYYYmm=figname))


CP_DIR = utils.get_cfg('CP', 'DATA_DIR')
CP_NAME = os.path.join('epoch{epoch:04d}-loss{loss:.4f}-val_loss{val_loss:.4f}.ckpt')

EARLY_STOPPING_CALLBACK = EarlyStopping(monitor='val_loss',
                                        min_delta=1e-4,
                                        patience=10,
                                        verbose=0,
                                        mode='auto',
                                        baseline=None,
                                        restore_best_weights=False
                                        )

FACTOR_PATH = utils.get_cfg('FACTOR', 'DATA_PATH')
if not os.path.exists(os.path.dirname(FACTOR_PATH)):
    os.makedirs(os.path.dirname(FACTOR_PATH))


def get_tensor(path, shuffle, buffer_sz=2 ** 10, batch_sz=2 ** 10):
    raw_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_dataset.map(parse_example)
    if shuffle == True:
        dataset = dataset.shuffle(buffer_sz)

    dataset = dataset.batch(batch_sz)
    return dataset


def pred(model, test_path):
    dataset_test = get_tensor(test_path, shuffle=False, buffer_sz=2 ** 10, batch_sz=2 ** 10)

    test_pred = model.predict(dataset_test)

    dtstrs = []
    codes = []
    raw_dataset_test = tf.data.TFRecordDataset(test_path)
    for dtstr, code in raw_dataset_test.map(parse_example_code_n_date):
        dtstrs.append(dtstr.numpy().decode('utf8'))
        codes.append(code.numpy().decode('utf8'))

    df = pd.DataFrame(test_pred, columns=['score'])
    df['date'] = pd.to_datetime(dtstrs)
    df['ticker'] = codes
    df.to_feather(FACTOR_PATH.format(YYYYmm=YYYYmm))
    save_outputstats(df, YYYYmm)


if __name__ == '__main__':
    alphanet = Alphanet()
    #     alphanet.compile(optimizer=RMSprop(lr=1e-4),loss='mse', metrics=['mse', 'mae'])
    alphanet.compile(optimizer=Adam(1e-5), loss='mse', metrics=['mse', 'mae'])

    train_cnt = 100

    for train_path, val_path, test_path in get_trainval_data_path_pairs():

        print(train_path, val_path, test_path)
        dataset_train = get_tensor(train_path, shuffle=True, buffer_sz=2 ** 10, batch_sz=2 ** 10)
        dataset_val = get_tensor(val_path, shuffle=False, buffer_sz=2 ** 10, batch_sz=2 ** 10)

        YYYYmm = os.path.basename(train_path).split('.')[0]
        cp_path = os.path.join(CP_DIR, YYYYmm, CP_NAME)
        if not os.path.exists(os.path.dirname(cp_path)):
            os.makedirs(os.path.dirname(cp_path))
        checkpoint_callback = ModelCheckpoint(cp_path, save_weights_only=True, save_best_only=False)

        if train_cnt > 0:
            history = alphanet.fit(dataset_train, validation_data=dataset_val, epochs=200,
                                   callbacks=[checkpoint_callback, EARLY_STOPPING_CALLBACK])
            train_cnt -= 1
            for l in alphanet.weights[4:8]:
                print(l)
            save_history(history, YYYYmm)
            if train_cnt == 0:
                break

            pred(alphanet, test_path)
