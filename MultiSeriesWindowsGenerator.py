import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


class MultiSeriesWindowsGenerator():
    def __init__(self, input_width, label_width, shift, batch_size, label_columns=[], GROUPBY=None,
                 regressor_columns=[], static_columns=[], DATE = "", LABELS = [""]):

        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = label_columns
        if len(label_columns) != 0:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}

        self.GROUPBY = GROUPBY
        self.regressor_columns = regressor_columns
        self.static_columns = static_columns
        self.DATE = DATE
        self.LABELS = LABELS

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Batch Size: {self.batch_size}',
            f'Label column name(s): {self.label_columns}',
            f'Additional Regressor column name(s): {self.regressor_columns}',
            f'GROUPBY column(s): {self.GROUPBY}'
        ])

    def preprocess_dataset(self, data: pd.DataFrame):
        try:
            if np.vstack(data.index).shape[1] != 1:
                data = data.reset_index()

            by = self.GROUPBY + [self.DATE]
            labels = self.label_columns + self.regressor_columns + self.static_columns
            data = data.set_index(by).unstack(-1)
            data.fillna(0, inplace=True)
            data = tf.stack([data[label] for label in labels], axis=-1)
            if data.ndim != 3:
                data = data[None, None, tf.newaxis]
        except Exception as e:
            print('Error while processing dataset', e)
        return data


    def update_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, norm: bool = False):
        # Store the raw data.
        self.train_df = self.preprocess_dataset(train_df)
        self.val_df = self.preprocess_dataset(val_df)
        self.test_df = self.preprocess_dataset(test_df)

        if norm:
            train_mean = tf.reduce_mean(self.train_df, axis=1, keepdims=True)
            train_std = tf.math.reduce_std(self.train_df, axis=1, keepdims=True)

            self.train_df = (self.train_df - train_mean) / train_std
            self.val_df = (self.val_df - train_mean) / train_std
            self.test_df = (self.test_df - train_mean) / train_std

            self.train_mean = train_mean
            self.train_std = train_std
            self.norm = norm

        labels = self.label_columns + self.regressor_columns + self.static_columns
        self.column_indices = {name: i for i, name in enumerate(labels)}

    def split_window(self, features: tf.Tensor):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, inputs, labels, model=None, plot_col=None, max_subplots=3):
        if not plot_col:
            plot_col = self.LABELS[0]
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} {"[normed]" if self.norm else ""}')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [Days]')
        plt.show()

    def make_cohort(self, data: np.array) -> tf.data.Dataset:
        data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size
        )
        ds = ds.map(self.split_window)
        return ds

    """
    Reference code from https://stackoverflow.com/questions/49994496/mixing-multiple-tf-data-dataset
    """


    def make_dataset(self, data: tf.Tensor) -> tf.data.Dataset:
        # num_cohorts = min(10, len(cluster_cohorts))
        # print(cluster, num_cohorts)
        def stack_windows(*windows):
            features = tf.concat([window[0] for window in windows], 0)
            labels = tf.concat([window[1] for window in windows], 0)
            return (features, labels)

        ds_list = tuple(self.make_cohort(data[i]) for i in range(len(data)))
        ds = tf.data.Dataset.zip(ds_list)
        ds = ds.map(stack_windows)
        ds = ds.unbatch()
        ds = ds.shuffle(10, seed=0)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    @property
    def train(self) -> tf.data.Dataset:
        return self.make_dataset(self.train_df)

    @property
    def val(self) -> tf.data.Dataset:
        return self.make_dataset(self.val_df)

    @property
    def test(self) -> tf.data.Dataset:
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting"""
        result = getattr(self, '_example', None)
        print('Number of train batches:', len(list(self.train.as_numpy_iterator())))
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


if __name__ == "__main__":

    data_list = []
    for i in range(0, 27):
        df = pd.read_csv(f"data/aggregated_individual_data_interpolation/interpolation/{i}_interpolated.csv",
                         index_col=0)
        df["subject_id"] = i + 1
        data_list.append(df)

    data = pd.concat(data_list)
    data.drop(["circumplex.arousal_std", "circumplex.valence_std", "mood_std", "activity_std", "date"], inplace=True,
              axis=1)
    df = data

    # TODO Fix this later in the preprocessing
    df.fillna(0, inplace=True)

    LABELS = ['mood']
    REGRESSORS = ['weekday', 'circumplex.arousal', 'circumplex.valence',
                  'activity', 'screen', 'call', 'sms', 'appCat.builtin',
                  'appCat.communication', 'appCat.entertainment', 'appCat.finance',
                  'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social',
                  'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']

    REGRESSORS = ['weekday']
    DATE = 'days'
    IN_STEPS = 24
    OUT_STEPS = 24
    GROUPBY = ['subject_id']
    BATCH_SIZE = 8

    n = len(df)
    train_series = df.groupby(GROUPBY, as_index=False, group_keys=False).apply(
        lambda x: x.iloc[:int(len(x) * 0.7)]).reset_index(drop=True)
    val_series = df.groupby(GROUPBY, as_index=False, group_keys=False).apply(
        lambda x: x.iloc[int(len(x) * 0.7):int(len(x) * 0.9)]).reset_index(drop=True)
    test_series = df.groupby(GROUPBY, as_index=False, group_keys=False).apply(
        lambda x: x.iloc[int(len(x) * 0.9):]).reset_index(drop=True)
    # train_series.shape, val_series.shape, test_series.shape


    # initialise window and feed split data into it
    w1 = MultiSeriesWindowsGenerator(input_width=IN_STEPS, label_width=OUT_STEPS, shift=OUT_STEPS,
                                     batch_size=BATCH_SIZE, label_columns=LABELS, GROUPBY=GROUPBY,
                                     regressor_columns=REGRESSORS, DATE=DATE, LABELS=LABELS)

    w1.update_datasets(train_series, val_series, test_series, norm=True)
    print(w1.train_df)

    print(w1.make_dataset(data=w1.train_df))


