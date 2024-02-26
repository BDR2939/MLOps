import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin


def process_data():
    path_to_data = "SKAB/"

    # benchmark files checking
    all_files = []
    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    valve1_data = {
        file.split("/")[-1]: pd.read_csv(
            file, sep=";", index_col="datetime", parse_dates=True
        )
        for file in all_files
        if "valve1" in file
    }

    # concatenate data(order in time series by sort_index)
    valve1_df = pd.concat(list(valve1_data.values()), axis=0).sort_index()
    valve1_df.drop_duplicates(inplace=True)

    valve2_data = {
        file.split("/")[-1]: pd.read_csv(
            file, sep=";", index_col="datetime", parse_dates=True
        )
        for file in all_files
        if "valve2" in file
    }

    # concatenate data(order in time series by sort_index)
    valve2_df = pd.concat(list(valve2_data.values()), axis=0).sort_index()
    valve2_df.drop_duplicates(inplace=True)

    other_anomaly_data = {
        file.split("/")[-1]: pd.read_csv(
            file, sep=";", index_col="datetime", parse_dates=True
        )
        for file in all_files
        if "other" in file
    }

    # concatenate data(order in time series by sort_index)
    other_anomaly_df = pd.concat(list(other_anomaly_data.values()), axis=0).sort_index()
    other_anomaly_df.drop_duplicates(inplace=True)

    valve1_X = valve1_df.drop(columns=["anomaly", "changepoint"])
    valve1_y = valve1_df.loc[:, ["anomaly", "changepoint"]]

    valve2_X = valve2_df.drop(columns=["anomaly", "changepoint"])
    valve2_y = valve2_df.loc[:, ["anomaly", "changepoint"]]

    other_anomaly_X = other_anomaly_df.drop(columns=["anomaly", "changepoint"])
    other_anomaly_y = other_anomaly_df.loc[:, ["anomaly", "changepoint"]]

    return {
        "valve1_X": valve1_X,
        "valve1_y": valve1_y,
        "valve2_X": valve2_X,
        "valve2_y": valve2_y,
        "other_anomaly_X": other_anomaly_X,
        "other_anomaly_y": other_anomaly_y,
    }


def get_single_df():
    path_to_data = "SKAB/"

    # benchmark files checking
    all_files = []
    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    all_data = {
        file.split("/")[-1]: pd.read_csv(
            file, sep=";", index_col="datetime", parse_dates=True
        )
        for file in all_files
    }

    # concatenate data(order in time series by sort_index)
    all_data = pd.concat(list(all_data.values()), axis=0).sort_index()

    all_data.fillna(0, inplace=True)
    all_data.drop_duplicates(inplace=True)

    all_data_X = all_data.drop(columns=["anomaly", "changepoint"])
    all_data_y = all_data.loc[:, ["anomaly", "changepoint"]]

    return all_data_X, all_data_y


def add_rolling_mean(df, time_diff, columns):
    for col in columns:
        df[f"{col}_rolling_mean"] = df[f"{col}"].rolling(time_diff).mean()
    return df


def add_rolling_skew(df, time_diff, columns):
    for col in columns:
        df[f"{col}_rolling_skew"] = df[f"{col}"].rolling(time_diff).skew()
    return df


def add_rolling_variance(df, time_diff, columns):
    for col in columns:
        df[f"{col}_rolling_variance"] = df[f"{col}"].rolling(time_diff).var()
    return df


def add_rolling_std(df, time_diff, columns):
    for col in columns:
        df[f"{col}_rolling_std"] = df[f"{col}"].rolling(time_diff).std()
    return df


def add_rolling_kurtosis(df, time_diff, columns):
    for col in columns:
        df[f"{col}_rolling_kurtosis"] = df[f"{col}"].rolling(time_diff).kurt()
    return df


def add_rolling_stats(df, time_diff, columns):
    df = add_rolling_mean(df.copy(), time_diff, columns)
    df = add_rolling_skew(df.copy(), time_diff, columns)
    df = add_rolling_variance(df.copy(), time_diff, columns)
    df = add_rolling_std(df.copy(), time_diff, columns)
    df = add_rolling_kurtosis(df.copy(), time_diff, columns)

    df.bfill(inplace=True)

    return df
