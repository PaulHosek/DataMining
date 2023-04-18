"""
Code adapted and modified from the following sources.
https://www.tensorflow.org/tutorials/structured_data/time_series
https://stackoverflow.com/questions/49994496/mixing-multiple-tf-data-dataset
https://medium.com/@kavyamalla/extending-tensorflows-window-generator-for-multiple-time-series-8b15eba57858
https://www.tensorflow.org/guide/keras/rnn
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

"""
The MultiSeriesWindowsGenerator class generates windowed datasets for multivariate time series data. 

The class works out the label column indices and window parameters based on the inputs. It then uses these parameters 
to generate windowed datasets from the input data using the make_dataset and make_cohort methods. The split_window 
method is used to split the windowed data into inputs and labels. 

The preprocess_dataset method is used to preprocess the input data and stack a boolean mask to indicate where NaN 
values are. The update_datasets method is used to update the stored raw data and normalize it if specified. """
class MultiSeriesWindowsGenerator():
    def __init__(self, input_width, label_width, shift, batch_size, label_columns=[], GROUPBY=None,
                 regressor_columns=[], static_columns=[], DATE = "", LABELS = [""]):

        self.train_max = None
        self.train_min = None
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
            by = self.GROUPBY + [self.DATE]  # [:-1]
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
        """
        Updates the stored training, validation, and test datasets, and optionally normalizes the data.
        Args:
            train_df: A pandas DataFrame containing training data.
            val_df: A pandas DataFrame containing validation data.
            test_df: A pandas DataFrame containing test data.
            norm: A boolean flag indicating whether to normalize the data. Defaults to False.
        Returns:
            None.
        """
        self.train_df = self.preprocess_dataset(train_df)
        self.val_df = self.preprocess_dataset(val_df)
        self.test_df = self.preprocess_dataset(test_df)

        if norm:
            train_min = tf.reduce_min(self.train_df, axis=1, keepdims=True)
            train_max = tf.reduce_max(self.train_df, axis=1, keepdims=True)
            self.train_df = 0.8 * (self.train_df - train_min) / (train_max - train_min) + 0.1
            self.val_df = 0.8 * (self.val_df - train_min) / (train_max - train_min) + 0.1
            self.test_df = 0.8 * (self.test_df - train_min) / (train_max - train_min) + 0.1

            self.train_min = train_min
            self.train_max = train_max
        self.norm = norm

        nan_mask = tf.math.is_nan(self.train_df)
        self.train_df = tf.where(nan_mask, tf.zeros_like(self.train_df), self.train_df)

        nan_mask = tf.math.is_nan(self.val_df)
        self.val_df = tf.where(nan_mask, tf.zeros_like(self.val_df), self.val_df)

        nan_mask = tf.math.is_nan(self.test_df)
        self.test_df = tf.where(nan_mask, tf.zeros_like(self.test_df), self.test_df)

        labels = self.label_columns + self.regressor_columns + self.static_columns
        self.column_indices = {name: i for i, name in enumerate(labels)}


        mask = np.array(tf.math.is_nan(self.test_df))
        # print(np.array(self.test_df[mask]))

    def split_window(self, features: tf.Tensor):
        """
        Splits a window of time series data into input and label windows, and returns them as two tensorflow Tensors.
        Args: features: A tensorflow Tensor with shape [batch_size, window_size, num_features].
        Returns: A tuple of two tensorflow Tensors: `inputs` with shape [batch_size, input_width, num_features]
         containing the input window, and `labels` with shape [batch_size, label_width, num_labels]
          containing the label window.

        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col=None, max_subplots=3, single=False):
        inputs, labels = self.example
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
                if single:
                    values = float(predictions[n, label_col_index])
                    x_values = self.label_indices[-1]
                else:
                    values = predictions[n, :, label_col_index]
                    x_values = self.label_indices

                plt.scatter(x_values, values, marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend(facecolor='white', framealpha=1, edgecolor='black')

        plt.xlabel('Time [Days]')
        plt.show()

    def make_cohort(self, data: np.array) -> tf.data.Dataset:
        """
        Creates a tensorflow Dataset from a numpy array of time series data, with a sliding window of `total_window_size`
        and a batch size of `batch_size`.
        Args:
            data: A numpy array of time series data
            with shape [num_samples, num_features].
        Returns:
            A tensorflow Dataset containing a sliding window of the
         time series data, split into input and label windows.
        """
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


    def make_dataset(self, data: tf.Tensor) -> tf.data.Dataset:
        """
        Creates a tensorflow dataset from input data.
        Args:
            data (tf.Tensor): The input data.
        Returns:
            tf.data.Dataset: The dataset with features and labels batches, ready for training.
        """
        def stack_windows(*windows):
            """
            Concatenates the input and label tensors in a window.
            Args:
                *windows (tf.Tensor): The tensors to concatenate.
            Returns:
                tf.Tensor: The concatenated tensors.
            """
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
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

    @property
    def test_data(self):
        """Get and cache all batches of `inputs, labels` for the test set"""
        result = getattr(self, '_test_data', None)
        if result is None:
            result = []
            for batch in self.test:
                result.append(batch)
            self._test_data = result
        return result


# if __name__ == "__main__":
#
#     # data_list = []
#     # for i in range(0, 27):
#     #     df = pd.read_csv(f"data/aggregated_individual_data_interpolation/interpolation/{i}_interpolated.csv",
#     #                      index_col=0)
#     #     df = pd.read_csv(f"data/aggregated_individual_data_interpolation/interpolation/{i}_interpolated.csv",
#     #                      index_col=0)
#     #     df["subject_id"] = i + 1
#     #     data_list.append(df)
#     #
#     # # Concatenate the data into a single dataset
#     # data = pd.concat(data_list)
#     # data.drop(["circumplex.arousal_std", "circumplex.valence_std", "mood_std", "activity_std"], inplace=True, axis=1)
#     df = pd.read_csv("data/all_data_aggr_nonan.csv", index_col=0)
#
#     LABELS = ['mood']
#     REGRESSORS = ['weekday', 'circumplex.arousal', 'circumplex.valence',
#                   'activity', 'screen', 'call', 'sms', 'appCat.builtin',
#                   'appCat.communication', 'appCat.entertainment', 'appCat.finance',
#                   'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social',
#                   'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
#
#     DATE = 'days'  # always correct
#     IN_STEPS = 14  # use 7 days
#     OUT_STEPS = 14  # to predict 1 day in the future
#     GROUPBY = ['subject_id']
#     BATCH_SIZE = 8
#
#     n = len(df)
#     train_series = df.groupby(GROUPBY, as_index=False, group_keys=False).apply(
#         lambda x: x.iloc[:int(len(x) * 0.7)]).reset_index(drop=True)
#     val_series = df.groupby(GROUPBY, as_index=False, group_keys=False).apply(
#         lambda x: x.iloc[int(len(x) * 0.7):int(len(x) * 0.9)]).reset_index(drop=True)
#     test_series = df.groupby(GROUPBY, as_index=False, group_keys=False).apply(
#         lambda x: x.iloc[int(len(x) * 0.9):]).reset_index(drop=True)
#
#     test_window = MultiSeriesWindowsGenerator(
#         input_width=IN_STEPS, label_width=OUT_STEPS, shift=1, batch_size=BATCH_SIZE, GROUPBY=GROUPBY,
#         label_columns=LABELS, regressor_columns=REGRESSORS, DATE=DATE, LABELS=LABELS)
#
#     test_window.update_datasets(train_series, val_series, test_series, norm=True)
#
#     lstm_model = tf.keras.models.Sequential([
#         tf.keras.layers.LSTM(128, return_sequences=True,activation="relu"),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.LSTM(128, return_sequences=False, activation="relu"),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))
#     ])
#
#     lstm_model = tf.keras.models.Sequential([
#         # Shape [batch, time, features] => [batch, time, lstm_units]
#         tf.keras.layers.LSTM(32, return_sequences=True),
#         # Shape => [batch, time, features]
#         tf.keras.layers.Dense(units=1)
#     ])
#
#     MAX_EPOCHS = 20
#
#
#     def compile_and_fit(model, window, patience=2):
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
#                                                           patience=patience,
#                                                           mode='min')
#
#         model.compile(loss=tf.keras.losses.MeanSquaredError(),
#                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                       metrics=[tf.keras.metrics.MeanAbsoluteError()])
#
#         history = model.fit(window.train, epochs=MAX_EPOCHS,
#                             validation_data=window.val,
#                             callbacks=[early_stopping])
#         return history
#
#     history = compile_and_fit(lstm_model, test_window)
#     val_performance = {}
#     performance = {}
#     val_performance['LSTM'] = lstm_model.evaluate(test_window.val)
#     performance['LSTM'] = lstm_model.evaluate(test_window.test, verbose=0)
#
#     test_window.plot(lstm_model)




