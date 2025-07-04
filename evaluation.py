#Script invoked by sagemaker_entrypoint.py
#Provides methods to evaluate the model during training using predetermined evaluation metrics like mae, mape etc.

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, f1_score, accuracy_score
from fnibot_model.fnibot_model import FNIBotModel


def print_metrics(name: str, metrics: dict):
    """
    Function to print metrics using the metrics dicitonary
    Paratmeters
    -----------
    name: string like
        Metric name
    metrics: dictionary like
        Dictionary containing the metrics and their values
    """
    for metric_name, metric_value in metrics.items():
        print(f"{name}__{metric_name}: {metric_value}")


def evaluate(test_df, model: FNIBotModel):
    """
    Function to calculate evaluation metrics
    Paratmeters
    -----------
    test_df: Data Frame like
        Input Data Frame for which predictions need to be made
    model: class like
        Model class that provides preprocessing and prediction capabilities
    
    Returns
    -------
    dictionary
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(test_df)
    y_true = model.get_target_data(test_df)

    if model.model_type == 'rate':
        results = calculate_rate_metrics(y_true, y_pred)
    elif model.model_type == 'approval':
        results = calculate_approval_metrics(y_true, y_pred)
    else:
        results = {}

    return results


def calculate_qual_metrics(y_true, y_pred, threshold=0.5):
    results = {
        "roc_auc_score": roc_auc_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred > threshold),
        "accuracy": accuracy_score(y_true, y_pred > threshold)
    }
    return results


def calculate_approval_metrics(y_true, y_pred):
    results = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true ))
    }
    return results