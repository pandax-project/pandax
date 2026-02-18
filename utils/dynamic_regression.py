import time
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def fit_linear_regression(x: List[float], y: List[float], new_factors: List[float]):
    """
    Fit a linear regression model to the data.

    Args:
        x (List[float]): Factor values.
        y (List[float]): Time in ms.
        new_factors (List[float]: New factor values to predict.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    model = LinearRegression()
    start_time = time.time()
    model.fit(x, y)
    end_time = time.time()
    print(f"Learnt a linear model in {end_time - start_time:.2f} seconds")
    print(f"R^2 score: {r2_score(y, model.predict(x)):.4f}")
    print(f"RMSE: {mean_squared_error(y, model.predict(x), squared=False):.4f}")
    print("MSE:", mean_squared_error(y, model.predict(x)))
    predictions = model.predict(np.array(new_factors).reshape(-1, 1))
    return predictions


def moving_average_with_padding(data, window_size=2):
    pad = window_size // 2
    padded = np.pad(data, (pad, pad), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")


def fit_linear_regression_smoothed(
    x: List[float], y: List[float], new_factors: List[float]
):
    """
    x (List[float]): Factor values.
    y (List[float]): Time in ms.
    new_factors (List[float]): New factor values to predict.
    """
    smoothed_x = moving_average_with_padding(x)
    smoothed_x = smoothed_x.reshape(-1, 1)
    model = LinearRegression()
    start_time = time.time()
    model.fit(smoothed_x, y)
    end_time = time.time()
    print(f"Learnt a smoothed linear model in {end_time - start_time:.2f} seconds")
    print(f"R^2 score: {r2_score(y, model.predict(smoothed_x)):.4f}")
    print(
        f"RMSE: {mean_squared_error(y, model.predict(smoothed_x), squared=False):.4f}"
    )
    print("MSE:", mean_squared_error(y, model.predict(smoothed_x)))
    predictions = model.predict(np.array(new_factors).reshape(-1, 1))
    return predictions
