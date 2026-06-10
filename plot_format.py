import matplotlib.pyplot as plt
from cycler import cycler


def setup_mpl():
    plt.rcParams["axes.prop_cycle"] = cycler(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    ) + cycler("linestyle", ["-", "--", "-.", ":", "--"])
    plt.style.use("plot_style.mplstyle")
