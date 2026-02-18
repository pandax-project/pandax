import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


# Apply moving average smoothing with padding
def moving_average_with_padding(data, window_size=2):
    pad = window_size // 2
    padded = np.pad(data, (pad, pad), mode="edge")  # pad by repeating edge values
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")


def linear_model():
    # wall_time in seconds
    # wall_time = np.array([19.65761185,21.1148262,22.2144126,24.2428779,26.1037349])
    # peak_mem in GB
    # peak_mem  = np.array([50,60,70,80, 90]).reshape(-1, 1)

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures

    data = """
10	6.071329117
30	5.877017975
50	6.810426712
70	7.080554962
90	6.864309311
110	8.290290833
130	9.05585289
150	7.734537125
170	8.114337921
190	7.878780365
210	7.935762405
230	9.073257446
250	8.755922318
270	9.295225143
290	8.814096451
310	9.427309036
    """

    cell = 11
    lines = data.strip().split("\n")
    peak_mem = []
    wall_time = []
    for line in lines:
        mem, time = map(float, line.split())
        peak_mem.append(mem)
        wall_time.append(time)
    peak_mem = np.array(peak_mem).reshape(-1, 1)
    wall_time = np.array(wall_time)

    # Split train/test
    peak_mem_train = peak_mem[:5]
    wall_time_train = wall_time[:5]
    # wall_time_train = moving_average_with_padding(wall_time_train, window_size=3)
    # wall_time_train = savgol_filter(wall_time_train, window_length=3, polyorder=1)
    peak_mem_test = peak_mem[5:]
    wall_time_test = wall_time[5:]

    # --- Linear Regression ---
    linear_model = LinearRegression()
    linear_model.fit(peak_mem_train, wall_time_train)
    wall_time_pred_linear = linear_model.predict(peak_mem_test)

    # --- Quadratic Regression ---
    poly = PolynomialFeatures(degree=2)
    peak_mem_train_poly = poly.fit_transform(peak_mem_train)
    peak_mem_test_poly = poly.transform(peak_mem_test)

    quad_model = LinearRegression()
    quad_model.fit(peak_mem_train_poly, wall_time_train)
    wall_time_pred_quad = quad_model.predict(peak_mem_test_poly)

    # --- Evaluation (optional) ---
    mse_linear = mean_squared_error(wall_time_test, wall_time_pred_linear)
    mse_quad = mean_squared_error(wall_time_test, wall_time_pred_quad)
    r2_linear = r2_score(wall_time_test, wall_time_pred_linear)
    r2_quad = r2_score(wall_time_test, wall_time_pred_quad)

    print(f"Linear MSE: {mse_linear:.3f}, R²: {r2_linear:.3f}")
    print(f"Quadratic MSE: {mse_quad:.3f}, R²: {r2_quad:.3f}")

    # --- Log-Log Regression ---
    log_peak_mem_train = np.log(peak_mem_train)
    log_wall_time_train = np.log(wall_time_train)
    log_peak_mem_test = np.log(peak_mem_test)
    np.log(wall_time_test)

    loglog_model = LinearRegression()
    loglog_model.fit(log_peak_mem_train, log_wall_time_train)
    log_wall_time_pred_loglog = loglog_model.predict(log_peak_mem_test)
    wall_time_pred_loglog = np.exp(log_wall_time_pred_loglog)

    # Evaluation for Log-Log Regression
    mse_loglog = mean_squared_error(wall_time_test, wall_time_pred_loglog)
    r2_loglog = r2_score(wall_time_test, wall_time_pred_loglog)
    print(f"Log-Log MSE: {mse_loglog:.3f}, R²: {r2_loglog:.3f}")

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(peak_mem_train, wall_time_train, color="blue", label="Training Data")
    plt.scatter(peak_mem_test, wall_time_test, color="green", label="Testing Data")

    # Sort test data for smooth lines
    sorted_test_indices = np.argsort(peak_mem_test.flatten())
    sorted_peak_mem_test = peak_mem_test[sorted_test_indices]

    # Predictions for plotting (sorted)
    wall_time_pred_linear_sorted = linear_model.predict(sorted_peak_mem_test)
    wall_time_pred_quad_sorted = quad_model.predict(
        poly.transform(sorted_peak_mem_test)
    )

    plt.plot(
        sorted_peak_mem_test,
        wall_time_pred_linear_sorted,
        color="red",
        label="Linear Fit",
    )
    plt.plot(
        sorted_peak_mem_test,
        wall_time_pred_quad_sorted,
        color="orange",
        linestyle="--",
        label="Quadratic Fit",
    )

    sorted_log_peak_mem_test = np.log(sorted_peak_mem_test)
    sorted_log_wall_time_pred_loglog = loglog_model.predict(sorted_log_peak_mem_test)
    sorted_wall_time_pred_loglog = np.exp(sorted_log_wall_time_pred_loglog)

    plt.plot(
        sorted_peak_mem_test,
        sorted_wall_time_pred_loglog,
        color="purple",
        linestyle=":",
        label="Log-Log Fit",
    )

    plt.xlabel("Peak Memory (GB)")
    plt.ylabel("Wall Time (s)")
    plt.title(
        f"Linear vs Quadratic Regression smoothing: Peak Memory vs Wall Time (Cell {cell})"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("comparison_linear_quadratic_regression.png")
    plt.show()

    # predicted_mem  = model.predict(new_wall_time)
    # print(f"\nPredicted peak_mem at {new_wall_time[0,0]} s wall-time: {predicted_mem[0]:.3f} GB")

    # # Try Ridge regression
    # ridge_model = Ridge(alpha=10.0)
    # ridge_model.fit(peak_mem, wall_time)
    # ridge_predicted_wall_time = ridge_model.predict(new_wall_time)
    # print(f"\nRidge Regression:")
    # print(f"Coefficient (slope):     {ridge_model.coef_[0]:.3f} GB/sec")
    # print(f"Intercept:              {ridge_model.intercept_:.3f} GB")
    # for n in range(100,210,10):
    #     new_wall_time = np.array([[n]])
    #     ridge_predicted_wall_time = ridge_model.predict(new_wall_time)
    #     print(f"{n},{ridge_predicted_wall_time[0]}")


def sinusoid_model():
    x = np.array([100, 110, 120, 130, 140, 150, 160])
    y = np.array([45.5, 48.3, 83.8, 95.1, 53.4, 48.1, 105])

    def model(x, C, A, omega, phi):
        return C + A * np.sin(omega * x + phi)

    # rough guesses: C≈mean, A≈half‐range, ω≈2π/period, φ=0
    p0 = [y.mean(), (y.max() - y.min()) / 2, 2 * np.pi / 60, 0]
    params, _ = curve_fit(model, x, y, p0=p0)
    C, A, omega, phi = params
    print(f"offset={C:.1f}, amp={A:.1f}, ω={omega:.3f}, φ={phi:.3f}")

    # new data
    new_x = np.array([170, 180, 190, 200])
    new_y = model(new_x, *params)
    print(f"Predicted values for new data: {new_y}")


def fft():
    x = np.array([100, 110, 120, 130, 140, 150, 160])
    y = np.array([45.5, 48.3, 83.8, 95.1, 53.4, 48.1, 105])
    delta_x = 10
    y0 = y - y.mean()  # remove mean

    Y = np.fft.fft(y0)
    freqs = np.fft.fftfreq(len(y0), delta_x)
    K = 2
    idx = np.argsort(np.abs(Y))[-K:]

    Y_sparse = np.zeros_like(Y)
    Y_sparse[idx] = Y[idx]
    np.fft.ifft(Y_sparse) + y.mean()

    # predict new data
    # new_x = np.array([170, 180, 190, 200])
    # new_y = np.interp(new_x, x, y_smooth.real)
    # print(f"Predicted values for new data: {new_y}")

    N = len(y0)
    Ak = 2 * np.abs(Y_sparse[idx]) / N  # amplitudes
    phik = np.angle(Y_sparse[idx])  # phases
    fk = freqs[idx]  # cycles per unit x
    C = y.mean()  # DC offset

    # 5) build an extrapolation function
    def extrapolate(x_new):
        """
        x_new: scalar or 1D array of factor values
        returns: predicted y_hat at those x_new
        """
        x_new = np.array(x_new, ndmin=1)
        t = (x_new - x[0]) / delta_x  # convert to sample units
        y_pred = np.full_like(t, fill_value=C, dtype=float)
        for A, f, phi in zip(Ak, fk, phik):
            y_pred += A * np.cos(2 * np.pi * f * t + phi)
        return y_pred

    # Example use: predict at 170,180,190,200
    x_test = np.array([170, 180, 190, 200])
    y_test = extrapolate(x_test)
    for xi, yi in zip(x_test, y_test):
        print(f"x={xi:3.0f} → ŷ={yi:.1f}")


# sinusoid_model()
def linear_model_all():
    # x = np.array([100,110,120,130,140,150,160]).reshape(-1, 1)
    y = np.array([45.5, 48.3, 83.8, 95.1, 53.4, 48.1, 105, 51.4, 72.4, 112, 50.2])
    x = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    # new data
    new_x = np.array([210, 220, 230]).reshape(-1, 1)
    new_y_pred = model.predict(new_x)
    print(f"Coefficient (slope):     {model.coef_[0]:.3f}")
    print(f"Intercept:              {model.intercept_:.3f}")
    print(f"Mean Squared Error:     {mse:.3f}")
    print(f"R^2 Score:              {r2:.3f}")
    print(f"Predicted values for new data: {new_y_pred.flatten()}")

    # try ridge
    ridge_model = Ridge(alpha=10000.0)
    ridge_model.fit(x, y)
    ridge_y_pred = ridge_model.predict(x)
    ridge_mse = mean_squared_error(y, ridge_y_pred)
    ridge_r2 = r2_score(y, ridge_y_pred)
    print("\nRidge Regression:")
    print(f"Coefficient (slope):     {ridge_model.coef_[0]:.3f}")
    print(f"Intercept:              {ridge_model.intercept_:.3f}")
    print(f"Mean Squared Error:     {ridge_mse:.3f}")
    print(f"R^2 Score:              {ridge_r2:.3f}")
    print(f"Predicted values for new data: {ridge_model.predict(new_x).flatten()}")

    # mean as the baseline
    baseline = np.mean(y)
    baseline_mse = mean_squared_error(y, np.full_like(y, baseline))
    print(f"\nBaseline Mean Squared Error: {baseline_mse:.3f}")
    print(f"Baseline R^2 Score: {r2_score(y, np.full_like(y, baseline)):.3f}")
    # rmse
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse:.3f}")

    # poly ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures

    poly_ridge_model = Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=4)),
            ("ridge_regression", Ridge(alpha=100.0)),
        ]
    )
    poly_ridge_model.fit(x, y)
    poly_ridge_y_pred = poly_ridge_model.predict(x)
    poly_ridge_mse = mean_squared_error(y, poly_ridge_y_pred)
    poly_ridge_r2 = r2_score(y, poly_ridge_y_pred)
    print("\nPolynomial Ridge Regression:")
    print(
        f"Coefficient (slope):     {poly_ridge_model.named_steps['ridge_regression'].coef_[0]:.3f}"
    )
    print(
        f"Intercept:              {poly_ridge_model.named_steps['ridge_regression'].intercept_:.3f}"
    )
    print(f"Mean Squared Error:     {poly_ridge_mse:.3f}")
    print(f"R^2 Score:              {poly_ridge_r2:.3f}")
    # Predicting new data with polynomial ridge regression
    poly_ridge_new_y_pred = poly_ridge_model.predict(new_x)
    print(f"Predicted values for new data: {poly_ridge_new_y_pred.flatten()}")

    # spline regression
    from scipy.interpolate import UnivariateSpline

    spline = UnivariateSpline(x.flatten(), y, k=3, s=100)
    spline_y_pred = spline(x.flatten())
    spline_mse = mean_squared_error(y, spline_y_pred)
    spline_r2 = r2_score(y, spline_y_pred)
    print("\nSpline Regression:")
    print(f"Mean Squared Error:     {spline_mse:.3f}")
    print(f"R^2 Score:              {spline_r2:.3f}")
    print(f"Predicted values for new data: {spline(new_x.flatten())}")

    # rbf kernel regression
    from scipy.interpolate import Rbf

    for s in range(1, 10, 1):
        rbf_model = Rbf(x.flatten(), y, function="gaussian", epsilon=10.0, smooth=s)
        rbf_y_pred = rbf_model(x.flatten())
        rbf_mse = mean_squared_error(y, rbf_y_pred)
        rbf_r2 = r2_score(y, rbf_y_pred)
        print(f"\nRBF Kernel Regression with epsilon={s}:")
        print(f"Mean Squared Error:     {rbf_mse:.3f}")
        print(f"R^2 Score:              {rbf_r2:.3f}")
        print(f"Predicted values for new data: {rbf_model(new_x.flatten())}")


linear_model()
