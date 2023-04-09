import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns

#
#
# def build_granger(id):
#     df = pd.read_csv("aggregated_individual_data/0_aggregated.csv", index_col=0, header=0, parse_dates=["date"])
#     # df["date"] = df["date"]
#     df = df.iloc[27:-1]
#     df['days'] = (df['date'] - df['date'].min()).dt.days.astype(int)
#     df.drop(["date", "weekday"], axis=1, inplace=True)
#     for col in df.columns:
#         df[col].fillna(df[col].mean(), inplace=True)
#
#     df = df.loc[:, (df != df.iloc[0]).any()]
#     # raise KeyboardInterrupt
#     granger_matrix = granger_causality_matrix(df[['days'] + list(df.columns)], list(df.columns))
#     return granger_matrix
#
#
#
# def granger_causality_matrix(data, variables, test="ssr_chi2test", maxlag=4):
#     df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
#
#     for c in df.columns:
#         for r in df.index:
#
#             # if too few values for granger skip column
#             if len(set(data[c])) <= 2 or len(set(data[r])) <= 2:
#                 continue
#
#             test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
#             p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
#             min_p_value = np.min(p_values)
#             df.loc[r, c] = min_p_value
#     df.columns = [var + '_x' for var in variables]
#     df.index = [var + '_y' for var in variables]
#     return df
def granger_causality_matrix(data, variables, test="ssr_chi2test", verbose=False, maxlag=12):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            # fill heatmap with smallest p-value found
            df.loc[r, c] = np.min([round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)])
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def build_granger_heatmap():
    df1 = pd.read_csv(f"aggregated_individual_data/{0}_aggregated.csv", index_col=0, parse_dates=["date"])
    df1['days'] = (df1['date'] - df1['date'].min()).dt.days.astype(int)
    df1.drop(["date"], axis=1, inplace=True)
    for i in range(1, 27):
        df2 = pd.read_csv(f"aggregated_individual_data/{0}_aggregated.csv", index_col=0, parse_dates=["date"])

        df2['days'] = (df2['date'] - df2['date'].min()).dt.days.astype(int)
        df2.drop(["date"], axis=1, inplace=True)
        df1 += df2
    df1.drop(["days", "weekday", "mood_std", "circumplex.valence_std", "circumplex.arousal_std", "activity_std",
              "appCat.weather"], axis=1, inplace=True)


    granger_matrix = granger_causality_matrix(df1, list(df1.columns), maxlag=1)
    print(granger_matrix)
    plt.figure(figsize=(12, 12))
    sns.heatmap(granger_matrix, cmap="Spectral_r")
    plt.show()
    granger_matrix = granger_causality_matrix(df1, list(df1.columns), maxlag=1)
    print(granger_matrix)
    plt.figure(figsize=(12, 12))
    sns.heatmap(granger_matrix, cmap="Spectral_r")
    plt.show()

if __name__ == "__main__":
    x = build_granger(1)
    plt.figure(figsize=(12, 12))
    sns.heatmap(x, cmap="Spectral_r")
    plt.show()
