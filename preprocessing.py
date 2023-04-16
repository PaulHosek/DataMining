import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


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

def aggregate_individual_data_per_reading(raw=pd.DataFrame, to_average=None ):
    """
    for the raw data of one individual a new df is generated with the variables as columns. Values are summed unless their variable name is 
    specified to be averaged in to_average.

    Parameters
    ----------
    INPUT
    raw : df
        DataFrame with the raw data
    to_average : array-like
        array specifiying which variables to average

    RETURNS
    processed : df
        DataFrame wiht the processed data
    """
    raw = raw.copy()
    # add column with date only
    raw['time'] = pd.to_datetime(raw.loc[:,'time'])
    raw.sort_values('time')

    # initialize new df with variables as columns
    vars = ['time', 'mood', 'circumplex.arousal', 'circumplex.valence',
       'activity', 'screen', 'call', 'sms', 'appCat.builtin',
       'appCat.communication', 'appCat.entertainment', 'appCat.finance',
       'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social',
       'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'sleep']
    processed = pd.DataFrame(columns = vars)

    measure_times = raw['time'].loc[raw['variable'] == 'mood']
    # add an date in the past for first measurement
    measure_times = pd.concat([pd.to_datetime(pd.Series(['2014-01-01 00:00:00.000'])), measure_times], ignore_index=True)

    # fill df
    # loop over times
    for i in range(len(measure_times)-1):
        processed.loc[i, 'time'] = measure_times[i+1]
        rows = (measure_times[i] < raw['time']) & (raw['time'] <= measure_times[i+1])

        # loop over variables
        for col in vars[1:-2]:

            # average values of given specified variables
            if col in to_average:
                # using mean of 1 value to get single value and not an array
                processed.loc[i, col] = raw.loc[(raw['time'] == measure_times[i]) & (raw['variable'] == col)].value.mean()

            
            # sum values for the other variables
            else:
                processed.loc[i, col] = raw.loc[rows & (raw['variable'] == col)].value.abs().sum()

    hours = dt.time(12)
    processed['time'] = pd.to_datetime(processed.loc[:,'time'])

    # add column for sleep / rest time
    for i, day in enumerate(processed['time'].dt.date.unique()):

        day_before = day - dt.timedelta(1)

        processed['sleep'].loc[processed['time'].dt.date == day] \
            = raw['time'].loc[(raw['time'] > dt.datetime.combine(day_before, hours)) & (raw['time'] < dt.datetime.combine(day, hours))] \
            .diff().max().total_seconds() / 3600

    # sort the df according to date
    processed['time'] = pd.to_datetime(processed.loc[:,'time'])
    processed.sort_values('time')

    # delete all data points where mood or screen are 0
    processed.drop(processed.loc[processed['mood'].isna() | (processed['screen'] == 0)].index, inplace=True)
    #processed.drop(processed.loc[processed['mood'].isna()].index, inplace=True)

    processed.reset_index(drop= True, inplace= True)

    # check that screen time does not exceed the time intervalls
    for i in range(1,len(processed)):
        Dt = processed['time'].loc[i] - processed['time'].loc[i-1]
        Dt = Dt.seconds
        if  Dt < processed['screen'].loc[i]:
            processed.iloc[i, 5:-1] = np.nan

    processed.insert(1, 'weekday', processed['time'].dt.weekday)

    return processed

def aggregate_per_day_from_measurement(raw=pd.DataFrame, to_average=None ):
    """
    for the aggregated data per measurement the values are aggregated per day. Summed unless the column name is 
    specified to be averaged in to_average.

    Parameters
    ----------
    INPUT
    raw : df
        DataFrame with the aggragated data per measurement
    to_average : array-like
        array specifiying which variables to average

    RETURNS
    processed : df
        DataFrame wiht the processed data
    """
    raw = raw.copy()

    # add column with date only
    raw['time'] = pd.to_datetime(raw.loc[:,'time'])
    raw['time'] = raw['time'].dt.date

    # initialize new df with variables as columns
    vars = raw.columns
    processed = pd.DataFrame(columns= vars)

    # fill df
    # loop over days
    for i, day in enumerate(raw['time'].unique()):
        processed.loc[i, 'time'] = day
        processed.loc[i, 'weekday'] = day.weekday()
        processed.loc[i, 'sleep'] = raw['sleep'].loc[raw['time'] == day].unique()[0]
        row = (raw['time'] == day)

        # loop over variables
        for col in vars[2:-2]:

            # average values of given specified variables
            if col in to_average:
                processed.loc[i, col] = raw.loc[row, col].mean()
                #processed.loc[i, col+'_std'] = raw.loc[row, col].value.std()

            
            # sum values for the other variables
            else:
                processed.loc[i, col] = raw.loc[row, col].abs().sum()

    # sort the df according to date
    processed['time'] = pd.to_datetime(processed.loc[:,'time'])
    processed.sort_values('time')

    # # drop rows without mood or or screen reading
    # processed.drop(processed.loc[processed['mood'].isna() & (processed['screen'] == 0)].index, inplace=True)

    processed.reset_index(drop= True, inplace= True)

    return processed