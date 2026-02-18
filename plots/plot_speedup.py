# Plot the speedup of the different backends.
# Note, you might have to tweak a few things to make this work, such as the CSV_PATH, FIGSIZE, LIN_THRESH, COLORS, etc.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ——— CONFIG ———
# CSV_PATH   = "dias_speedups.csv"
CSV_PATH = "ds_speedups.csv"
BACKENDS = ["cuDF", "DIAS", "pandaX"]
BASELINE = "pandas"
if "ds" in CSV_PATH:
    FIGSIZE = (4, 4)
else:
    FIGSIZE = (8, 4)
LIN_THRESH = 5.0  # symlog linear region up to 5×
# colors roughly matching your GEMM palette:
COLORS = ["gray", "#ffd285", "royalblue", "royalblue", "lightgray"]
# ——— END CONFIG ———


def main():
    # 1) load & compute speedup
    df = pd.read_csv(CSV_PATH, index_col=0)
    speedup = pd.DataFrame(
        df[BASELINE].values[:, None] / df[BACKENDS], index=df.index, columns=BACKENDS
    )

    # 2) style tweaks
    # mpl.style.use("seaborn-whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.8,
            "grid.linestyle": "--",
            "grid.color": "0.85",
            "grid.linewidth": 0.5,
        }
    )

    # 3) plot bars manually and calculate geometric means
    # Calculate figure width based on number of benchmarks
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    x = np.arange(len(speedup))
    if "ds" in CSV_PATH:
        width = 0.2  # Fixed width that looks good regardless of number of backends
    else:
        width = 0.2  # Fixed width that looks good regardless of number of backends

    # First calculate geometric means for all backends
    geomeans = {}
    for i, be in enumerate(BACKENDS):
        vals = np.array(speedup[be].values)
        valid_vals = vals[vals > 0]
        if len(valid_vals) > 0:
            geomean = np.exp(np.log(valid_vals).mean())
            geomeans[be] = geomean

    # Plot bars with updated labels that include geometric means
    for i, be in enumerate(BACKENDS):
        vals = speedup[be].values
        gmean_text = f" (GMean: {geomeans.get(be, 0):.2f})" if be in geomeans else ""
        bars = ax.bar(  # noqa: F841
            x + i * width,
            vals,
            width=width,
            label=f"{be}{gmean_text}",
            color=COLORS[i],
            edgecolor="none",
        )

    # Add subtle geometric mean lines
    for i, be in enumerate(BACKENDS):
        if be in geomeans:
            # Use darker yellow for DIAS line visibility
            line_color = "#e6a83e" if be == "DIAS" else COLORS[i]
            ax.axhline(
                geomeans[be], color=line_color, linestyle="--", linewidth=0.8, alpha=0.6
            )

    # 4) symlog y-axis
    # ax.set_yscale("symlog", linthreshy=LIN_THRESH, linscaley=1.0)
    ax.set_yscale("symlog")

    # 5) baseline line at y=1
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1)

    # 6) labels & legend (mimic your GEMM placement)
    ax.set_xticks(x + width * (len(BACKENDS) - 1) / 2)
    ax.set_xticklabels(speedup.index, rotation=30, ha="right")
    ax.set_ylabel("Speedup", labelpad=-10)

    # Layout legend based on dataset
    if "ds" in CSV_PATH:
        ax.legend(frameon=False, loc="best", ncol=1)  # Vertical layout
    else:
        ax.legend(
            frameon=False, loc="upper center", ncol=len(BACKENDS)
        )  # Horizontal layout

    if "ds" in CSV_PATH:
        plt.savefig("ds_speedup_chart.pdf", dpi=300, bbox_inches="tight")
    else:
        plt.savefig("dias_speedup_chart.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
