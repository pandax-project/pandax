import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

gpu_data = """
100	0	90.5
110	0	88.1
120	0	105
130	0	84.2
140	0	99.7
150	0	132
160	0	101
170	0	84.9
180	0	84.1
190	0	93.1
200	0	94.1

100	1	124
110	1	137
120	1	184
130	1	163
140	1	167
150	1	232
160	1	177
170	1	189
180	1	192
190	1	212
200	1	212

100	2	1.47
110	2	1.6
120	2	2.36
130	2	1.72
140	2	2.06
150	2	2.22
160	2	1.56
170	2	2.07
180	2	1.87
190	2	2.72
200	2	1.57

100	3	0.254
110	3	0.249
120	3	0.434
130	3	0.251
140	3	0.28
150	3	0.417
160	3	0.281
170	3	0.246
180	3	0.283
190	3	0.287
200	3	0.313

100	4	130
110	4	157
120	4	226
130	4	168
140	4	237
150	4	294
160	4	216
170	4	258
180	4	223
190	4	242
200	4	238

100	5	227
110	5	263
120	5	289
130	5	299
140	5	357
150	5	583
160	5	391
170	5	379
180	5	392
190	5	413
200	5	507

100	6	91.2
110	6	71.5
120	6	75.3
130	6	65.8
140	6	70.8
150	6	106
160	6	69.2
170	6	64.6
180	6	68.4
190	6	66.5
200	6	64.9

100	7	18.8
110	7	19.1
120	7	23
130	7	19.6
140	7	19
150	7	25.5
160	7	20.6
170	7	21.7
180	7	18.7
190	7	18.5
200	7	17.5

100	8	39.8
110	8	40.7
120	8	49.2
130	8	38.8
140	8	61.2
150	8	56.5
160	8	41
170	8	44.8
180	8	40.7
190	8	40.4
200	8	34.9

100	9	26.4
110	9	26.5
120	9	27
130	9	26.1
140	9	26.7
150	9	31.3
160	9	26.9
170	9	33.1
180	9	26
190	9	26.6
200	9	26

100	10	12.1
110	10	11.3
120	10	12.6
130	10	11.4
140	10	11.7
150	10	12.3
160	10	11.2
170	10	14.8
180	10	11.6
190	10	12.5
200	10	12.1

100	11	44.3
110	11	43.3
120	11	57.7
130	11	42.2
140	11	41.7
150	11	43
160	11	41.9
170	11	54.5
180	11	42.1
190	11	45.2
200	11	44.5
"""

cpu_data = """
100	0	2.99
110	0	4.92
120	0	4.9
130	0	4.6
140	0	4.4
150	0	3.94
160	0	3.07
170	0	4.17
180	0	3.32
190	0	4.02
200	0	4.38

100	1	34.5
110	1	66.2
120	1	46.2
130	1	52.3
140	1	49
150	1	77.7
160	1	54.2
170	1	84.4
180	1	61.3
190	1	63.8
200	1	81.5

100	2	0.0989
110	2	0.156
120	2	0.163
130	2	0.102
140	2	0.17
150	2	0.148
160	2	0.107
170	2	0.147
180	2	0.109
190	2	0.127
200	2	0.175

100	3	0.0634
110	3	0.0973
120	3	0.0658
130	3	0.0632
140	3	0.103
150	3	0.228
160	3	0.0892
170	3	0.31
180	3	0.0963
190	3	0.103
200	3	0.0963

100	4	14.3
110	4	24
120	4	18.6
130	4	18.1
140	4	24.7
150	4	24.2
160	4	21.9
170	4	26.3
180	4	21.9
190	4	26.2
200	4	25.3

100	5	19.5
110	5	31.6
120	5	23.7
130	5	25.7
140	5	44.5
150	5	50.9
160	5	39.3
170	5	57.5
180	5	42.8
190	5	47.7
200	5	56.5

100	6	40.4
110	6	51.5
120	6	60.3
130	6	81.3
140	6	80.4
150	6	103
160	6	76
170	6	125
180	6	97.8
190	6	82.3
200	6	89.3

100	7	43.3
110	7	68.5
120	7	77
130	7	97.7
140	7	73.5
150	7	112
160	7	79.1
170	7	74.3
180	7	85.4
190	7	104
200	7	103

100	8	51.6
110	8	72.3
120	8	76.2
130	8	75.7
140	8	83.4
150	8	125
160	8	91
170	8	152
180	8	115
190	8	115
200	8	106

100	9	57.2
110	9	78.5
120	9	68.7
130	9	86.2
140	9	83.4
150	9	150
160	9	154
170	9	95.9
180	9	124
190	9	114
200	9	186

100	10	2.84
110	10	3.56
120	10	4.26
130	10	5.72
140	10	4
150	10	5.96
160	10	5.39
170	10	4.77
180	10	5.52
190	10	6.61
200	10	5.98

100	11	48.7
110	11	62.1
120	11	73.8
130	11	63.7
140	11	75.3
150	11	103
160	11	83.2
170	11	85.9
180	11	132
190	11	174
200	11	97.2
"""

# TODO
data = gpu_data


def smooth(values, window_size=3):
    if len(values) < window_size:
        return values
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = window_size // 2
    return np.concatenate(
        [
            values[:pad],
            smoothed,
            values[-pad:] if window_size % 2 == 1 else values[-pad + 1 :],
        ]
    )


# Parse and group data
groups = {}
for line in data.strip().splitlines():
    if not line.strip():
        continue
    x, group, y = line.split()
    x, group, y = int(x), int(group), float(y)
    groups.setdefault(group, []).append((x, y))

# Create subplots
num_groups = len(groups)
fig, axs = plt.subplots(num_groups, 1, figsize=(8, 4 * num_groups), sharex=True)
if num_groups == 1:
    axs = [axs]

# Plot each group and predict for factor = 1000
print("Predictions at factor = 1000:")
for ax, (group, values) in zip(axs, sorted(groups.items(), key=lambda g: int(g[0]))):
    values.sort()
    xs, ys = zip(*values)
    xs = np.array(xs)
    ys = np.array(ys)
    smoothed_ys = smooth(ys, window_size=3)

    ax.plot(xs, ys, marker="o", label="Original", alpha=0.4)
    ax.plot(xs, smoothed_ys, marker="o", label="Smoothed", linewidth=2)

    increasing_steps = sum(
        1 for i in range(len(smoothed_ys) - 1) if smoothed_ys[i + 1] > smoothed_ys[i]
    )
    monotonicity = increasing_steps / (len(smoothed_ys) - 1)
    avg = np.mean(ys)

    if monotonicity >= 0.7:
        model = LinearRegression()
        # model.fit(xs.reshape(-1, 1), smoothed_ys)
        model.fit(xs.reshape(-1, 1), ys)
        preds = model.predict(xs.reshape(-1, 1))
        y_pred = model.predict(np.array([[1000]]))[0]
        ax.plot(xs, preds, label="Linear Fit", color="red", linestyle="--")
        print(
            f"Group {group}: Linear prediction at factor 1000 = {y_pred:.2f}, monotonicity = {monotonicity:.2f}"
        )
    else:
        y_pred = avg
        ax.axhline(y=avg, color="purple", linestyle="--", label="Average")
        print(
            f"Group {group}: Average prediction at factor 1000 = {y_pred:.2f}, monotonicity = {monotonicity:.2f}"
        )

    ax.set_title(f"cell {group} GPU")
    ax.set_ylabel("value")
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel("Factor (X)")
plt.tight_layout()
plt.savefig("cell_gpu_original_fit_plots.png", dpi=300)
