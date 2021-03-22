import tensorflow as tf
from layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, ReLU, Dropout, BatchNormalization, AvgPool1D, AvgPool2D, MaxPool1D, \
    MaxPool2D


class Alphanet(Model):
    def __init__(self):
        super().__init__()
        # layer1
        # layer1 sub1
        self.ts_stddev10 = Ts_stddev(10)
        self.ts_stddev10_bn = BatchNormalization(axis=-1)

        self.ts_mean10 = Ts_mean(10)
        self.ts_mean10_bn = BatchNormalization(axis=-1)

        self.ts_ret10 = Ts_ret(10)
        self.ts_ret10_bn = BatchNormalization(axis=-1)

        self.ts_delaylinear10 = Ts_delaylinear(10)
        self.ts_delaylinear10_bn = BatchNormalization(axis=-1)

        self.ts_cov10 = Ts_cov(10)
        self.ts_cov10_bn = BatchNormalization(axis=-1)

        self.ts_corr10 = Ts_corr(10)
        self.ts_corr10_bn = BatchNormalization(axis=-1)

        self.ts_zscore10 = Ts_zscore(10)
        self.ts_zscore10_bn = BatchNormalization(axis=-1)

        # layer2
        # layer2 sub1
        self.ts_stddev10_bn_tsmean3 = AvgPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_stddev10_bn_tsmean3_bn = BatchNormalization()
        self.ts_stddev10_bn_tsmax3 = MaxPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_stddev10_bn_tsmax3_bn = BatchNormalization()

        self.ts_mean10_bn_tsmean3 = AvgPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_mean10_bn_tsmean3_bn = BatchNormalization()
        self.ts_mean10_bn_tsmax3 = MaxPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_mean10_bn_tsmax3_bn = BatchNormalization()

        self.ts_ret10_bn_tsmean3 = AvgPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_ret10_bn_tsmean3_bn = BatchNormalization()
        self.ts_ret10_bn_tsmax3 = MaxPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_ret10_bn_tsmax3_bn = BatchNormalization()

        self.ts_delaylinear10_bn_tsmean3 = AvgPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_delaylinear10_bn_tsmean3_bn = BatchNormalization()
        self.ts_delaylinear10_bn_tsmax3 = MaxPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_delaylinear10_bn_tsmax3_bn = BatchNormalization()

        self.ts_cov10_bn_tsmean3 = AvgPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_cov10_bn_tsmean3_bn = BatchNormalization()
        self.ts_cov10_bn_tsmax3 = MaxPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_cov10_bn_tsmax3_bn = BatchNormalization()

        self.ts_corr10_bn_tsmean3 = AvgPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_corr10_bn_tsmean3_bn = BatchNormalization()
        self.ts_corr10_bn_tsmax3 = MaxPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_corr10_bn_tsmax3_bn = BatchNormalization()

        self.ts_zscore10_bn_tsmean3 = AvgPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_zscore10_bn_tsmean3_bn = BatchNormalization()
        self.ts_zscore10_bn_tsmax3 = MaxPool2D(pool_size=(1, 3), strides=1)  # strides=1?
        self.ts_zscore10_bn_tsmax3_bn = BatchNormalization()

        # layer3

        self.ts_stddev10_bn_flatten = Flatten()
        self.ts_stddev10_bn_tsmean3_bn_flatten = Flatten()
        self.ts_stddev10_bn_tsmax3_bn_flatten = Flatten()

        self.ts_mean10_bn_flatten = Flatten()
        self.ts_mean10_bn_tsmean3_bn_flatten = Flatten()
        self.ts_mean10_bn_tsmax3_bn_flatten = Flatten()

        self.ts_ret10_bn_flatten = Flatten()
        self.ts_ret10_bn_tsmean3_bn_flatten = Flatten()
        self.ts_ret10_bn_tsmax3_bn_flatten = Flatten()

        self.ts_delaylinear10_bn_flatten = Flatten()
        self.ts_delaylinear10_bn_tsmean3_bn_flatten = Flatten()
        self.ts_delaylinear10_bn_tsmax3_bn_flatten = Flatten()

        self.ts_cov10_bn_flatten = Flatten()
        self.ts_cov10_bn_tsmean3_bn_flatten = Flatten()
        self.ts_cov10_bn_tsmax3_bn_flatten = Flatten()

        self.ts_corr10_bn_flatten = Flatten()
        self.ts_corr10_bn_tsmean3_bn_flatten = Flatten()
        self.ts_corr10_bn_tsmax3_bn_flatten = Flatten()

        self.ts_zscore10_bn_flatten = Flatten()
        self.ts_zscore10_bn_tsmean3_bn_flatten = Flatten()
        self.ts_zscore10_bn_tsmax3_bn_flatten = Flatten()

        # layer4
        self.dense30 = Dense(30, activation='relu')
        self.dropout30 = Dropout(0.5)
        self.dense301 = Dense(30, activation='relu')
        self.dropout301 = Dropout(0.5)

        # output layer
        self.dense1 = Dense(1, activation='linear')

    def call(self, inputs):
        # layer1
        ts_stddev10 = self.ts_stddev10(inputs)
        ts_stddev10_bn = self.ts_stddev10_bn(ts_stddev10)

        ts_mean10 = self.ts_mean10(inputs)
        ts_mean10_bn = self.ts_mean10_bn(ts_mean10)

        ts_ret10 = self.ts_ret10(inputs)
        ts_ret10_bn = self.ts_ret10_bn(ts_ret10)

        ts_delaylinear10 = self.ts_delaylinear10(inputs)
        ts_delaylinear10_bn = self.ts_delaylinear10_bn(ts_delaylinear10)

        ts_cov10 = self.ts_cov10(inputs)
        ts_cov10_bn = self.ts_cov10_bn(ts_cov10)

        ts_corr10 = self.ts_corr10(inputs)
        ts_corr10_bn = self.ts_corr10_bn(ts_corr10)

        ts_zscore10 = self.ts_zscore10(inputs)
        ts_zscore10_bn = self.ts_zscore10_bn(ts_zscore10)

        # layer2
        ts_stddev10_bn_tsmean3 = self.ts_stddev10_bn_tsmean3(ts_stddev10_bn)
        ts_stddev10_bn_tsmean3_bn = self.ts_stddev10_bn_tsmean3_bn(ts_stddev10_bn_tsmean3)
        ts_stddev10_bn_tsmax3 = self.ts_stddev10_bn_tsmax3(ts_stddev10_bn)
        ts_stddev10_bn_tsmax3_bn = self.ts_stddev10_bn_tsmax3_bn(ts_stddev10_bn_tsmax3)

        ts_mean10_bn_tsmean3 = self.ts_mean10_bn_tsmean3(ts_mean10_bn)
        ts_mean10_bn_tsmean3_bn = self.ts_mean10_bn_tsmean3_bn(ts_mean10_bn_tsmean3)
        ts_mean10_bn_tsmax3 = self.ts_mean10_bn_tsmax3(ts_mean10_bn)
        ts_mean10_bn_tsmax3_bn = self.ts_mean10_bn_tsmax3_bn(ts_mean10_bn_tsmax3)

        ts_ret10_bn_tsmean3 = self.ts_ret10_bn_tsmean3(ts_ret10_bn)
        ts_ret10_bn_tsmean3_bn = self.ts_ret10_bn_tsmean3_bn(ts_ret10_bn_tsmean3)
        ts_ret10_bn_tsmax3 = self.ts_ret10_bn_tsmax3(ts_ret10_bn)
        ts_ret10_bn_tsmax3_bn = self.ts_ret10_bn_tsmax3_bn(ts_ret10_bn_tsmax3)

        ts_delaylinear10_bn_tsmean3 = self.ts_delaylinear10_bn_tsmean3(ts_delaylinear10_bn)
        ts_delaylinear10_bn_tsmean3_bn = self.ts_delaylinear10_bn_tsmean3_bn(ts_delaylinear10_bn_tsmean3)
        ts_delaylinear10_bn_tsmax3 = self.ts_delaylinear10_bn_tsmax3(ts_delaylinear10_bn)
        ts_delaylinear10_bn_tsmax3_bn = self.ts_delaylinear10_bn_tsmax3_bn(ts_delaylinear10_bn_tsmax3)

        ts_cov10_bn_tsmean3 = self.ts_cov10_bn_tsmean3(ts_cov10_bn)
        ts_cov10_bn_tsmean3_bn = self.ts_cov10_bn_tsmean3_bn(ts_cov10_bn_tsmean3)
        ts_cov10_bn_tsmax3 = self.ts_cov10_bn_tsmax3(ts_cov10_bn)
        ts_cov10_bn_tsmax3_bn = self.ts_cov10_bn_tsmax3_bn(ts_cov10_bn_tsmax3)

        ts_corr10_bn_tsmean3 = self.ts_corr10_bn_tsmean3(ts_corr10_bn)
        ts_corr10_bn_tsmean3_bn = self.ts_corr10_bn_tsmean3_bn(ts_corr10_bn_tsmean3)
        ts_corr10_bn_tsmax3 = self.ts_corr10_bn_tsmax3(ts_corr10_bn)
        ts_corr10_bn_tsmax3_bn = self.ts_corr10_bn_tsmax3_bn(ts_corr10_bn_tsmax3)

        ts_zscore10_bn_tsmean3 = self.ts_zscore10_bn_tsmean3(ts_zscore10_bn)
        ts_zscore10_bn_tsmean3_bn = self.ts_zscore10_bn_tsmean3_bn(ts_zscore10_bn_tsmean3)
        ts_zscore10_bn_tsmax3 = self.ts_zscore10_bn_tsmax3(ts_zscore10_bn)
        ts_zscore10_bn_tsmax3_bn = self.ts_zscore10_bn_tsmax3_bn(ts_zscore10_bn_tsmax3)

        # layer3
        ts_stddev10_bn_flatten = self.ts_stddev10_bn_flatten(ts_stddev10_bn)
        ts_stddev10_bn_tsmean3_bn_flatten = self.ts_stddev10_bn_tsmean3_bn_flatten(ts_stddev10_bn_tsmean3_bn)
        ts_stddev10_bn_tsmax3_bn_flatten = self.ts_stddev10_bn_tsmax3_bn_flatten(ts_stddev10_bn_tsmax3_bn)

        ts_mean10_bn_flatten = self.ts_mean10_bn_flatten(ts_mean10_bn)
        ts_mean10_bn_tsmean3_bn_flatten = self.ts_mean10_bn_tsmean3_bn_flatten(ts_mean10_bn_tsmean3_bn)
        ts_mean10_bn_tsmax3_bn_flatten = self.ts_mean10_bn_tsmax3_bn_flatten(ts_mean10_bn_tsmax3_bn)

        ts_ret10_bn_flatten = self.ts_ret10_bn_flatten(ts_ret10_bn)
        ts_ret10_bn_tsmean3_bn_flatten = self.ts_ret10_bn_tsmean3_bn_flatten(ts_ret10_bn_tsmean3_bn)
        ts_ret10_bn_tsmax3_bn_flatten = self.ts_ret10_bn_tsmax3_bn_flatten(ts_ret10_bn_tsmax3_bn)

        ts_delaylinear10_bn_flatten = self.ts_delaylinear10_bn_flatten(ts_delaylinear10_bn)
        ts_delaylinear10_bn_tsmean3_bn_flatten = self.ts_delaylinear10_bn_tsmean3_bn_flatten(
            ts_delaylinear10_bn_tsmean3_bn)
        ts_delaylinear10_bn_tsmax3_bn_flatten = self.ts_delaylinear10_bn_tsmax3_bn_flatten(
            ts_delaylinear10_bn_tsmax3_bn)

        ts_cov10_bn_flatten = self.ts_cov10_bn_flatten(ts_cov10_bn)
        ts_cov10_bn_tsmean3_bn_flatten = self.ts_cov10_bn_tsmean3_bn_flatten(ts_cov10_bn_tsmean3_bn)
        ts_cov10_bn_tsmax3_bn_flatten = self.ts_cov10_bn_tsmax3_bn_flatten(ts_cov10_bn_tsmax3_bn)

        ts_corr10_bn_flatten = self.ts_corr10_bn_flatten(ts_corr10_bn)
        ts_corr10_bn_tsmean3_bn_flatten = self.ts_corr10_bn_tsmean3_bn_flatten(ts_corr10_bn_tsmean3_bn)
        ts_corr10_bn_tsmax3_bn_flatten = self.ts_corr10_bn_tsmax3_bn_flatten(ts_corr10_bn_tsmax3_bn)

        ts_zscore10_bn_flatten = self.ts_zscore10_bn_flatten(ts_zscore10_bn)
        ts_zscore10_bn_tsmean3_bn_flatten = self.ts_zscore10_bn_tsmean3_bn_flatten(ts_zscore10_bn_tsmean3_bn)
        ts_zscore10_bn_tsmax3_bn_flatten = self.ts_zscore10_bn_tsmax3_bn_flatten(ts_zscore10_bn_tsmax3_bn)

        concatenated = tf.keras.layers.concatenate(
            [ts_stddev10_bn_flatten, ts_stddev10_bn_tsmean3_bn_flatten, ts_stddev10_bn_tsmax3_bn_flatten,
             ts_mean10_bn_flatten, ts_mean10_bn_tsmean3_bn_flatten, ts_mean10_bn_tsmax3_bn_flatten,
             ts_ret10_bn_flatten, ts_ret10_bn_tsmean3_bn_flatten, ts_ret10_bn_tsmax3_bn_flatten,
             ts_delaylinear10_bn_flatten, ts_delaylinear10_bn_tsmean3_bn_flatten, ts_delaylinear10_bn_tsmax3_bn_flatten,
             ts_cov10_bn_flatten, ts_cov10_bn_tsmean3_bn_flatten, ts_cov10_bn_tsmax3_bn_flatten,
             ts_corr10_bn_flatten, ts_corr10_bn_tsmean3_bn_flatten, ts_corr10_bn_tsmax3_bn_flatten,
             ts_zscore10_bn_flatten, ts_zscore10_bn_tsmean3_bn_flatten, ts_zscore10_bn_tsmax3_bn_flatten
             ], axis=-1)

        # layer4
        output = self.dense30(concatenated)
        output = self.dropout30(output)
        output = self.dense301(output)
        output = self.dropout301(output)

        # output layer
        output = self.dense1(output)
        return output
