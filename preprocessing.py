import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def interpolate_aggr(id, in_path="data/aggregated_individual_data_interpolation/raw",
                     out_path="data/aggregated_individual_data_interpolation/interpolation"):
    """Use time-interpolation on dataset to fill np.NaN values. Beginning and end are filled with 0s."""
    df = pd.read_csv(f"{in_path}/{id}_aggregated.csv", parse_dates=["date"], index_col="date")
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.interpolate(inplace=True, method="time")
    df.reset_index(inplace=True)
    df['days'] = (df['date'] - df['date'].min()).dt.days.astype(int)
    df.fillna(0)  # fill beginning and end with 0 if cannot interpolate
    df.to_csv(f"{out_path}/{id}_interpolated.csv")
