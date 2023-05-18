
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
import random

from tqdm import tqdm

from xgboost import XGBRegressor, XGBRanker
#from sklearn.xgboost import XGBRegressor

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, HistGradientBoostingClassifier, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, ndcg_score
from sklearn.utils import resample
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, GroupKFold
import lightgbm
import flaml
import joblib

def import_df(filename):
    """
    Add target column to dataframe and remove date-time column.
    :param filename: "name.csv"
    :return: pd.DF
    """
    temp_df = pd.read_csv(f'data/{filename}')
    temp_df.drop(['Unnamed: 0',"date_time"], axis=1, inplace=True)
    temp_df['target'] = temp_df['click_bool'] + 4 * temp_df['booking_bool']
    temp_df['target'].loc[temp_df['target'].isna()] = int(0)
    return temp_df

def undersample(train_df, drop_cols = ['click_bool', 'gross_bookings_usd',
                                                'booking_bool', 'position', 'target']):
    """
    Undersample non-bookings
    :param train_df:
    :param drop_cols:
    :return:
    """
    X_under, y_under = resample(
        train_df.drop(drop_cols, axis=1).loc[train_df['target'] < 1],
        train_df['target'].loc[train_df['target'] < 1]
        , n_samples=1 * sum(train_df['target'] >= 1), replace=False)

    return X_under, y_under

def balance(train_df):
    X_bal = train_df.drop(['click_bool', 'gross_bookings_usd', 'booking_bool', 'position', 'target'], axis=1)
    y_bal = train_df['target']
    return X_bal, y_bal

def drop_comp_cols(df):
    inv_cols = [f"comp{i}_inv" for i in np.arange(1, 9)]
    rate_cols = [f"comp{i}_rate" for i in np.arange(1, 9)]
    rate_perc_cols = [f"comp{i}_rate_percent_diff" for i in np.arange(1, 9)]
    return df.drop(columns=inv_cols + rate_cols + rate_perc_cols)

def build_groups(df):
    return df.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

def fit_automodel(x_df,y_df,groups, parameters = {}, **kwargs):
    default_parameters = {
        "task": "rank",
        "groups": groups,
        "metric": "ndcg",
        "estimator_list": ["lgbm"]
    }

    # Update default parameters with specified parameters
    default_parameters.update(parameters)
    model = flaml.AutoML()
    model.fit(x_df.drop(columns=['srch_id']), y_df, **default_parameters, **kwargs)
    return model


def fit_manualmodel(x_df,y_df,groups,manual_model, **kwargs):
    """
    Fit a non-autotuned model.
    Provide additional arguments to the fit method as if you are using to it directly:
    e.g., fit_manualmodel(..., epochs=10, batch_size=32)
    """
    manual_model.fit(x_df, y_df, group=groups,**kwargs)

def save_mod(model, name):
    joblib.dump(model,f"{name}.plk")


def load_mod(name):
    return joblib.load(f"{name}.plk")


def eval_ndcg(test_df, model):
    temp = test_df.drop(columns=np.setdiff1d(test_df.columns,model.feature_name_))
    test_df['pred_score'] = model.predict(temp)

    test_df['pred_rank'] = test_df.groupby('srch_id')['pred_score'].rank(ascending=False).astype(int)
    test_ids = test_df["srch_id"].unique()
    print('MSE', mean_squared_error(test_df['target'], test_df['pred_score']))

    # scoring with ndcg
    mean_ndcg = 0
    counter = 0
    pbar = tqdm(test_ids)  # Initialize the tqdm progress bar with test_ids
    for id in pbar:
        if len(test_df['target'].loc[test_df['srch_id'] == id]) > 1:
            ndcg = ndcg_score([test_df['target'].loc[test_df['srch_id'] == id].astype(int).to_numpy()],
                              [test_df['pred_score'].loc[test_df['srch_id'] == id].to_numpy()], k=5)

        mean_ndcg += ndcg
        if ndcg < 1.0:
            counter += 1
        pbar.set_description(f"Mean NDCG: {mean_ndcg / (counter + 1):.4f}")  # Update the progress bar description

    mean_ndcg = mean_ndcg / len(test_ids)
    return mean_ndcg

def make_submission(model):
    comp_data = pd.read_csv('data/test_set_VU_DM.csv')
    comp_data.drop('date_time', axis=1, inplace=True)
    comp_data['prediction'] = model.predict(comp_data.drop(['srch_id'], axis=1))

    comp_data.sort_values(['srch_id', 'prediction'], axis=0,
                          inplace=True, ignore_index=True, ascending=[True, False])
    filename = 'data/predictions/prediction' + str(datetime.now()) + '.csv'
    comp_data[['srch_id', 'prop_id']].to_csv(filename, index=False)
    print(f"Submission saved under {filename}")

