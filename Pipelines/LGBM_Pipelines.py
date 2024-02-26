# libraries importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
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


def LGBM_base():
    model = lgb.LGBMClassifier()

    ml_pipe = Pipeline(steps=[("Classifier", model)])

    return ml_pipe


def LGBM_valves_pipeline(X):
    model = lgb.LGBMClassifier()

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
                "Outlier capping",
                Winsorizer(capping_method="quantiles", tail="both", fold=3),
            ),
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


def PCA_winsorizer_pipeline(model):
    ml_pipe = Pipeline(
        steps=[
            (
                "Outlier capping",
                Winsorizer(capping_method="quantiles", tail="both", fold=3),
            ),
            ("PCA", PCA(n_components="mle", svd_solver="full")),
            ("Classifier", model),
        ]
    )

    return ml_pipe


def LGBM_single_step_pipeline(X, step):
    model = lgb.LGBMClassifier()

    steps_dict = {
        "Add Rolling Stats": FunctionTransformer(
            add_rolling_stats,
            kw_args={"time_diff": "1min", "columns": X.columns},
        ),
        "Outlier capping": Winsorizer(capping_method="quantiles", tail="both", fold=3),
        "Features Selection": SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=42), cv=2
        ),
        "PCA": PCA(n_components="mle", svd_solver="full"),
    }

    ml_pipe = Pipeline(
        steps=[
            (step, steps_dict[step]),
            ("Classifier", model),
        ]
    )

    return ml_pipe


def LGBM_ablation_study_pipeline(X, steps):
    model = lgb.LGBMClassifier()

    steps_dict = {
        "Add Rolling Stats": FunctionTransformer(
            add_rolling_stats,
            kw_args={"time_diff": "1min", "columns": X.columns},
        ),
        "Outlier capping": Winsorizer(capping_method="quantiles", tail="both", fold=3),
        "Features Selection": SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=42), cv=2
        ),
        "PCA": PCA(n_components="mle", svd_solver="full"),
    }

    steps = [(step, steps_dict[step]) for step in steps]

    ml_pipe = Pipeline(steps=steps + [("Classifier", model)])

    return ml_pipe


def benchmarks(X_train, y_train, X_test, y_test, pipeline):
    start_time = time.perf_counter()

    y_pred = pipeline.fit(X_train, y_train).predict(X_test)

    end_time = time.perf_counter()

    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    TN, FP, FN, TP = conf_matrix.ravel()

    # Fall out or false positive rate FAR false alarm rate
    FPR = FP / (FP + TN)
    # False negative rate MAR missing alarm rate
    FNR = FN / (TP + FN)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    set_config(display="text")

    benchmarks_data = pd.DataFrame(
        {
            "steps": [[name for name, func in pipeline.steps]],
            "FPR (FAR)": FPR,
            "FNR (MAR)": FNR,
            "ACC": ACC,
        }
    )

    y_pred_series = pd.Series(y_pred, index=y_test.index, name="prediction")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

    ax1.plot(y_test, label="Ground Truth")
    ax1.set_title("Ground Truth (y_test)")
    ax1.legend()
    ax1.set_ylabel("Anomaly")

    ax2.plot(y_pred_series, label="Prediction", color="orange")
    ax2.set_title("Predictions (y_pred)")
    ax2.legend()
    ax2.set_ylabel("Anomaly")

    ax2.set_xlabel("Date")
    plt.tight_layout()
    plt.show()

    return benchmarks_data
