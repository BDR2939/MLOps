# libraries importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config
from sklearn.compose import make_column_selector


from data_processing.process_data import add_rolling_stats
from algorithms.Conv_AE import Conv_AE

from feature_engine.selection import (
    SmartCorrelatedSelection,
    SelectBySingleFeaturePerformance,
    RecursiveFeatureAddition,
)
from feature_engine.outliers import Winsorizer


N_STEPS = 120


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=N_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


def test_train_split(df_X, df_y):
    size_train = int(df_X.shape[0] * 0.8)
    size_test = df_X.shape[0] - size_train
    x_train = df_X[:size_train]
    y_train = df_y[:size_train].anomaly
    x_test = df_X[-size_test:]
    y_test = df_y[-size_test:].anomaly
    return x_train, y_train, x_test, y_test


def eval_results(X, y, Q, steps=[]):
    model = Conv_AE()

    x_train, y_train, x_test, y_test = test_train_split(X, y)

    steps_dict = {
        "Add Rolling Stats": FunctionTransformer(
            add_rolling_stats,
            kw_args={"time_diff": "1min", "columns": X.columns},
        ),
        "Scaling": StandardScaler(),
        "Outlier capping": Winsorizer(capping_method="quantiles", tail="both", fold=3),
        "Features Selection": SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=42), cv=2
        ),
        "PCA": PCA(n_components="mle", svd_solver="full"),
    }

    for step in steps:
        func = steps_dict[step]
        func.fit(x_train, y_train)
        x_train = func.transform(x_train)
        x_test = func.transform(x_test)

    combination = steps + ["Classifier"]

    x_train_steps = create_sequences(x_train, N_STEPS)
    x_test_steps = create_sequences(x_test, N_STEPS)

    model.fit(x_train_steps)

    # results predicting
    residuals = pd.Series(
        np.sum(
            np.mean(np.abs(x_train_steps - model.predict(x_train_steps)), axis=1),
            axis=1,
        )
    )
    UCL = residuals.quantile(Q)

    # train prediction
    cnn_residuals = pd.Series(
        np.sum(
            np.mean(np.abs(x_train_steps - model.predict(x_train_steps)), axis=1),
            axis=1,
        )
    )

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data = cnn_residuals > UCL
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(x_train_steps) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    yhat_train = pd.Series(data=0, index=np.arange(len(x_train)))
    yhat_train.iloc[anomalous_data_indices] = 1

    # test prediction
    cnn_residuals = pd.Series(
        np.sum(
            np.mean(np.abs(x_test_steps - model.predict(x_test_steps)), axis=1), axis=1
        )
    )

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data = cnn_residuals > UCL
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(x_test_steps) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    yhat_test = pd.Series(data=0, index=np.arange(len(x_test)))
    yhat_test.iloc[anomalous_data_indices] = 1

    conf_matrix = metrics.confusion_matrix(y_test, yhat_test)

    TN, FP, FN, TP = conf_matrix.ravel()

    # Fall out or false positive rate FAR false alarm rate
    FPR = FP / (FP + TN)
    # False negative rate MAR missing alarm rate
    FNR = FN / (TP + FN)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    benchmarks_data = pd.DataFrame(
        {"steps": [combination], "FPR (FAR)": FPR, "FNR (MAR)": FNR, "ACC": ACC},
        index=[0],
    )
    
    yhat_test.index = y_test.index
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

    ax1.plot(y_test, label='Ground Truth')
    ax1.set_title('Ground Truth (y_test)')
    ax1.legend()
    ax1.set_ylabel('Anomaly')

    ax2.plot(yhat_test, label='Prediction', color='orange')
    ax2.set_title('Predictions (y_pred)')
    ax2.legend()
    ax2.set_ylabel('Anomaly')

    ax2.set_xlabel('Date')
    plt.tight_layout()
    plt.show()

    return benchmarks_data


def Conv_AE_base():
    model = Conv_AE()

    ml_pipe = Pipeline(steps=[("Classifier", model)])

    return ml_pipe


def Conv_AE_Example_pipeline(X):
    model = Conv_AE()

    ml_pipe = Pipeline(
        steps=[
            (
                "Add Rolling Stats",
                FunctionTransformer(
                    add_rolling_stats,
                    kw_args={"time_diff": "1min", "columns": X.columns},
                ),
            ),
            (
                "Scaler",
                StandardScaler(),
            ),
            ("PCA", PCA(n_components="mle", svd_solver="full")),
            (
                "Features Selection",
                SelectBySingleFeaturePerformance(
                    RandomForestClassifier(random_state=42), cv=2
                ),
            ),
            ("Classifier", model),
        ]
    )

    return ml_pipe


def Conv_AE_Example_pipeline2():
    model = Conv_AE()

    ml_pipe = Pipeline(
        steps=[
            (
                "Outlier capping",
                Winsorizer(capping_method="quantiles", tail="both", fold=3),
            ),
            ("PCA", PCA(n_components="mle", svd_solver="full")),
            (
                "Features Selection",
                SelectBySingleFeaturePerformance(
                    RandomForestClassifier(random_state=42), cv=2
                ),
            ),
            ("Classifier", model),
        ]
    )

    return ml_pipe
