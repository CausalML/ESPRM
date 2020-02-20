import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from make_regret_graphs import get_method_name, baseline, filter_by_method

results_dir = "synthetic_results"
file_prefix = "linear_linear"


def square_error(theta_1, theta_2):
    theta_1_np = np.array([float(v) for v in theta_1.split(",")])
    theta_2_np = np.array([float(v) for v in theta_2.split(",")])
    theta_1_norm = float((theta_1_np ** 2).sum() ** 0.5)
    theta_2_norm = float((theta_2_np ** 2).sum() ** 0.5)
    x1 = theta_1_np / theta_1_norm
    x2 = theta_2_np / theta_2_norm
    return float(((x1 - x2) ** 2).sum())


def make_graphs():
    df = pd.read_csv("%s/%s.csv" % (results_dir, file_prefix))
    df = filter_by_method(df)
    err_list = []
    method_list = []
    baseline_key_dict = {}
    for i in range(len(df)):
        err = square_error(df["theta"][i], df["optimal_theta"][i])
        err_list.append(err)
        method = get_method_name(df["weights"][i])
        method_list.append(method)
        if method == baseline:
            key = (df["num_train"][i], df["rep"][i])
            baseline_key_dict[key] = err
    baseline_list = []
    for i in range(len(df)):
        key = (df["num_train"][i], df["rep"][i])
        baseline_list.append(baseline_key_dict[key])

    df["baseline"] = baseline_list
    df["theta_square_error"] = err_list
    df["se_decrease"] = (df["baseline"] - df["theta_square_error"])
    df["Method"] = method_list

    # make graph
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set(xscale="log")
    ax.set(yscale="log")
    plot = sns.lineplot(x="num_train", y="theta_square_error", style="Method",
                        hue_order=["ERM", "ESPRM"],
                        style_order=["ERM", "ESPRM"],
                        hue="Method", ci=95, data=df,
                        palette=["k", "r"])
    ax.set_title("Convergence of parameter MSE", fontsize=30)
    ax.set_xlabel("Training Set Size", fontsize=24)
    ax.set_ylabel("MSE", fontsize=24)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    plot.legend(prop={'size': 16}, loc="upper right")
    fig.subplots_adjust(bottom=0.16)
    fig.savefig("%s/theta_mse_%s.png" % (results_dir, file_prefix))

    # make graph
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set(xscale="log")
    # ax.set(yscale="log")
    plot = sns.lineplot(x="num_train", y="se_decrease_scaled", style="Method",
                        hue_order=["ERM", "ESPRM"],
                        style_order=["ERM", "ESPRM"],
                        hue="Method", ci=95, data=df,
                        palette=["k", "r"])
    ax.set_title("Mean Decrease in Square Error", fontsize=30)
    ax.set_xlabel("Training Set Size", fontsize=24)
    ax.set_ylabel("Square Error Decrease", fontsize=24)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    plot.legend(prop={'size': 16}, loc="upper right")
    fig.subplots_adjust(bottom=0.16)
    fig.savefig("%s/theta_mse_decraese_%s.png" % (results_dir, file_prefix))


if __name__ == "__main__":
    make_graphs()
