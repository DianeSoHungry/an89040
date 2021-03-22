import tensorflow as tf
from tensorflow.keras.layers import Layer


def replace_anamly(x, axix, n_std):
    x_mean = tf.math.reduce_mean(x, axis=axix, keepdims=True)
    x_std = tf.math.reduce_std(x, axis=axix, keepdims=True)
    x_ceil = (x_mean + x_std * n_std)
    x_floor = (x_mean - x_std * n_std)
    x = tf.where(x < x_ceil, x, x_ceil)
    x = tf.where(x > x_floor, x, x_floor)
    return x


class Ts_mean(Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs_mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), -1)
        inputs_mean = tf.repeat(inputs_mean, 30, axis=-1)
        inputs_std = (tf.reduce_mean((inputs - inputs_mean) ** 2, axis=-1)) ** (1 / 2)
        inputs_std = tf.expand_dims(inputs_std, -1)
        inputs_std = tf.repeat(inputs_std, 30, axis=-1)
        inputs = (inputs - inputs_mean) / inputs_std

        slides = list()
        for i in range(self.window_size):
            slides.append(inputs[..., i:(inputs.shape[-1] - self.window_size + i) + 1])
        win_mean = tf.reduce_mean(slides, axis=0)
        #         print(tf.expand_dims(win_mean,axis=-1).shape)
        return tf.expand_dims(win_mean, axis=-1)


#         return win_mean

class Ts_delaylinear(Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs_mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), -1)
        inputs_mean = tf.repeat(inputs_mean, 30, axis=-1)
        inputs_std = (tf.reduce_mean((inputs - inputs_mean) ** 2, axis=-1)) ** (1 / 2)
        inputs_std = tf.expand_dims(inputs_std, -1)
        inputs_std = tf.repeat(inputs_std, 30, axis=-1)
        inputs = (inputs - inputs_mean) / inputs_std

        slides = list()
        for i in range(self.window_size):
            slides.append(inputs[..., i:(inputs.shape[-1] - self.window_size + i) + 1] * (i + 1))
        slides = tf.stack(slides, axis=0)

        delaylinear = tf.reduce_mean(slides, axis=0) / (sum(range(i)) + 1)
        #         print(tf.expand_dims(win_mean,axis=-1).shape)
        return tf.expand_dims(delaylinear, axis=-1)


class Ts_ret(Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs_mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), -1)
        inputs_mean = tf.repeat(inputs_mean, 30, axis=-1)
        inputs_std = (tf.reduce_mean((inputs - inputs_mean) ** 2, axis=-1)) ** (1 / 2)
        inputs_std = tf.expand_dims(inputs_std, -1)
        inputs_std = tf.repeat(inputs_std, 30, axis=-1)
        inputs = (inputs - inputs_mean) / inputs_std

        slides = list()
        for i in range(self.window_size - 1, inputs.shape[-1]):
            slides.append(inputs[..., i] - inputs[..., i - self.window_size + 1])
        ret = tf.stack(slides, axis=-1)

        #         print(tf.expand_dims(win_mean,axis=-1).shape)
        return tf.expand_dims(ret, axis=-1)


class Ts_stddev(Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs_mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), -1)
        inputs_mean = tf.repeat(inputs_mean, 30, axis=-1)
        inputs_std = (tf.reduce_mean((inputs - inputs_mean) ** 2, axis=-1)) ** (1 / 2)
        inputs_std = tf.expand_dims(inputs_std, -1)
        inputs_std = tf.repeat(inputs_std, 30, axis=-1)
        inputs = (inputs - inputs_mean) / inputs_std
        slides = list()
        for i in range(self.window_size):
            slides.append(inputs[..., i:(inputs.shape[-1] - self.window_size + i) + 1])

        win_mean = tf.reduce_mean(slides, axis=0)
        total_loss = [(s - win_mean) ** 2 for s in slides]
        ret = tf.reduce_mean(total_loss, axis=0) ** (1 / 2)
        return tf.expand_dims(ret, axis=-1)


#         return ret

class Ts_zscore(Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs_mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), -1)
        inputs_mean = tf.repeat(inputs_mean, 30, axis=-1)
        inputs_std = (tf.reduce_mean((inputs - inputs_mean) ** 2, axis=-1)) ** (1 / 2)
        inputs_std = tf.expand_dims(inputs_std, -1)
        inputs_std = tf.repeat(inputs_std, 30, axis=-1)
        inputs = (inputs - inputs_mean) / inputs_std
        slides = list()
        for i in range(self.window_size):
            slides.append(inputs[..., i:(inputs.shape[-1] - self.window_size + i) + 1])

        win_mean = tf.reduce_mean(slides, axis=0)
        total_loss = [(s - win_mean) ** 2 for s in slides]
        std = tf.reduce_mean(total_loss, axis=0) ** (1 / 2)
        ret = win_mean / (std + 1e-8)
        ret = replace_anamly(ret, -1, 3)
        return tf.expand_dims(ret, axis=-1)


class Ts_cov(Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def build(self, input_shape):

        pass

    def call(self, inputs):
        inputs_mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), -1)
        inputs_mean = tf.repeat(inputs_mean, 30, axis=-1)
        inputs_std = (tf.reduce_mean((inputs - inputs_mean) ** 2, axis=-1)) ** (1 / 2)
        inputs_std = tf.expand_dims(inputs_std, -1)
        inputs_std = tf.repeat(inputs_std, 30, axis=-1)
        inputs = (inputs - inputs_mean) / inputs_std

        new_rows = []
        rows = inputs.shape[-2]
        for i in range(rows):
            for j in range(i, rows):
                new_rows.append(tf.multiply(inputs[:, i, :], inputs[:, j, :]))
        inputs = tf.stack(new_rows, axis=1)

        slides = list()
        for i in range(self.window_size):
            slides.append(inputs[..., i:(inputs.shape[-1] - self.window_size + i) + 1])

        win_mean = tf.reduce_mean(slides, axis=0)
        #         print(tf.expand_dims(win_mean,axis=-1).shape)
        return tf.expand_dims(win_mean, axis=-1)


class Ts_corr(Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def build(self, input_shape):

        pass

    def call(self, inputs):
        inputs_mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), -1)
        inputs_mean = tf.repeat(inputs_mean, 30, axis=-1)
        inputs_std = (tf.reduce_mean((inputs - inputs_mean) ** 2, axis=-1)) ** (1 / 2)
        inputs_std = tf.expand_dims(inputs_std, -1)
        inputs_std = tf.repeat(inputs_std, 30, axis=-1)
        inputs = (inputs - inputs_mean) / inputs_std

        # 得到cov
        new_rows = []
        rows = inputs.shape[-2]
        for i in range(rows):
            for j in range(i, rows):
                new_rows.append(tf.multiply(inputs[:, i, :], inputs[:, j, :]))
        cov = tf.stack(new_rows, axis=1)

        slides = list()
        for i in range(self.window_size):
            slides.append(cov[..., i:(inputs.shape[-1] - self.window_size + i) + 1])

        cov = tf.expand_dims(tf.reduce_mean(slides, axis=0), axis=-1)

        # 得到std_i
        new_rows = []
        rows = inputs.shape[-2]
        for i in range(rows):
            for j in range(i, rows):
                new_rows.append(tf.multiply(inputs[:, i, :], inputs[:, i, :]))
        std_i = tf.stack(new_rows, axis=1)

        slides = list()
        for i in range(self.window_size):
            slides.append(std_i[..., i:(inputs.shape[-1] - self.window_size + i) + 1])

        std_i = tf.expand_dims((tf.reduce_mean(slides, axis=0) + 1e-8) ** (1 / 2), axis=-1)

        # 得到std_j
        new_rows = []
        rows = inputs.shape[-2]
        for i in range(rows):
            for j in range(i, rows):
                new_rows.append(tf.multiply(inputs[:, j, :], inputs[:, j, :]))
        std_j = tf.stack(new_rows, axis=1)

        slides = list()
        for i in range(self.window_size):
            slides.append(std_j[..., i:(inputs.shape[-1] - self.window_size + i) + 1])

        std_j = tf.expand_dims((tf.reduce_mean(slides, axis=0) + 1e-8) ** (1 / 2), axis=-1)

        # 得到corr
        corr = tf.divide(cov, std_i)
        corr = tf.divide(corr, std_j)
        return corr
