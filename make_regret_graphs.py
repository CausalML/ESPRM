import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


results_dir = "synthetic_results"
file_names = ("linear_linear", "linear_flexible",
              "quadratic_linear", "quadratic_flexible")
baseline = "ERM"


def get_scenario_name(s):
    if s == "linear":
        return "LinearScenario"
    elif s == "quadratic":
        return "QuadraticScenario"


def get_policy_name(s):
    if s == "linear":
        return "LinearPolicy"
    elif s == "flexible":
        return "FlexiblePolicy"


def get_method_name(weights_str):
    if "Deep" in weights_str:
        return "ESPRM"
    elif weights_str == "Unweighted" or weights_str == "None":
        return "ERM"
    elif "poly" in weights_str:
        degree = int(weights_str.split("_")[1])
        if degree != 3:
            return None
        return "Poly(%d)" % degree
    elif "rks" in weights_str:
        n = int(weights_str.split("_")[1])
        if n != 64:
            return None
        return "RBF(%d)" % n


def add_baseline_column(df):
    # parse df to get dict of results
    method_names = []
    results_dict = defaultdict(dict)
    for i in range(len(df)):
        num_train = df["num_train"][i]
        rep = df["rep"][i]
        key = (num_train, rep)
        weights = df["weights"][i]
        method_name = get_method_name(weights)
        method_names.append(method_name)
        optimal_val = df["optimal_policy_val"][i]
        if key in results_dict:
            assert(results_dict[key]["optimal"] == optimal_val)
        else:
            results_dict[key]["optimal"] = optimal_val
        results_dict[key][method_name] = df["test_policy_val"][i]
    df["Method"] = method_names

    # add baseline column
    baseline_results = np.zeros(len(df))
    for i in range(len(df)):
        num_train = df["num_train"][i]
        rep = df["rep"][i]
        key = (num_train, rep)
        baseline_results[i] = results_dict[key][baseline]
    df["baseline"] = baseline_results


def filter_by_method(df):
    weights = df["weights"]
    idx = []
    for i in range(len(weights)):
        method_name = get_method_name(weights[i])
        if method_name is not None:
            idx.append(i)
    return df.iloc[idx, :].reset_index(drop=True)


def drop_fgmm(df):
    weights = df["weights"]
    idx = []
    for i in range(len(weights)):
        method_name = get_method_name(weights[i])
        if method_name in ("ERM", "ESPRM"):
            idx.append(i)
    return df.iloc[idx, :].reset_index(drop=True)


def randomize_data_per_n(df):
    method_n_policy_vals = defaultdict(lambda: defaultdict(dict))
    optimal_vals = defaultdict(dict)

    for i in range(len(df)):
        method = df["Method"][i]
        n = df["num_train"][i]
        optimal_val = df["optimal_policy_val"][i]
        policy_val = df["test_policy_val"][i]
        rep = df["rep"][i]
        method_n_policy_vals[n][method][rep] = policy_val
        if method == baseline:
            optimal_vals[n][rep] = optimal_val

    n_list = []
    method_list = []
    optimal_val_list = []
    policy_val_list = []
    n_range = sorted(set(list(optimal_vals.keys())))
    for n in n_range:
        all_methods = sorted(set(list(method_n_policy_vals[n].keys())))
        num_reps = len(optimal_vals[n])
        idx = list(np.random.choice(list(range(num_reps)), size=num_reps))
        for rep in idx:
            for method in all_methods:
                n_list.append(n)
                method_list.append(method)
                optimal_val_list.append(optimal_vals[n][rep])
                policy_val_list.append(method_n_policy_vals[n][method][rep])
    return pd.DataFrame({"num_train": n_list,
                         "Method": method_list,
                         "optimal_policy_val": optimal_val_list,
                         "test_policy_val": policy_val_list})


def get_regret_transformed_data_bootstrapping(df, num_bootstrap=1000):
    for i in range(num_bootstrap):
        df_random = randomize_data_per_n(df)
        dft = get_regret_transformed_data(df_random, i)
        if i == 0:
            df_bs = dft
        else:
            df_bs = pd.concat([df_bs, dft], axis=0, ignore_index=True)
    return df_bs


def get_regret_transformed_data(df, bootstrap_i=0):
    method_n_policy_vals = defaultdict(list)
    optimal_vals = defaultdict(list)
    for i in range(len(df)):
        method = df["Method"][i]
        n = df["num_train"][i]
        optimal_val = df["optimal_policy_val"][i]
        policy_val = df["test_policy_val"][i]
        method_n_policy_vals[(method, n)].append(policy_val)
        if method == baseline:
            optimal_vals[n].append(optimal_val)

    n_list = []
    method_list = []
    regret_improvement_list = []

    for (method, n), policy_vals in method_n_policy_vals.items():
        mean_optimal = float(np.mean(optimal_vals[n]))
        mean_method = float(np.mean(policy_vals))
        mean_baseline = float(np.mean(method_n_policy_vals[(baseline, n)]))
        regret_improvement = 100.0 - 100.0 * ((mean_optimal - mean_method)
                                              / (mean_optimal - mean_baseline))
        n_list.append(n)
        method_list.append(method)
        regret_improvement_list.append(regret_improvement)
    bi_list = [bootstrap_i for _ in range(len(n_list))]
    return pd.DataFrame({"num_train": n_list,
                         "Method": method_list,
                         "bi": bi_list,
                         "regret_perc_improvement": regret_improvement_list})




def main():
    for f_name in file_names:
        path = "%s/%s.csv" % (results_dir, f_name)
        fgmm_path = "%s/fgmm_%s.csv" % (results_dir, f_name)
        df_1 = pd.read_csv(path, index_col=0)
        df_2 = pd.read_csv(fgmm_path, index_col=0)
        df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
        df = filter_by_method(df)
        df = drop_fgmm(df)
        add_baseline_column(df)
        df["baseline_regret"] = df["optimal_policy_val"] - df["baseline"]
        df["baseline_diff"] = df["test_policy_val"] - df["baseline"]
        b_d = df["baseline_diff"]
        b_r = df["baseline_regret"]
        df["norm_baseline_diff"] = 100.0 * (b_d / b_r)
        df["norm_baseline_diff"].fillna(0.0)

        dft = get_regret_transformed_data_bootstrapping(df, num_bootstrap=1000)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set(xscale="log")
        plot = sns.lineplot(x="num_train", y="regret_perc_improvement",
                            hue="Method", style="Method", data=dft, ci="sd",
                            palette=["k", "r"])
        scenario_name = get_scenario_name(f_name.split("_")[-2])
        policy_name = get_policy_name(f_name.split("_")[-1])
        ax.set_title("%s, %s" % (scenario_name, policy_name), fontsize=30)
        ax.set_xlabel("Training Set Size", fontsize=24)
        ax.set_ylabel("RMRR", fontsize=24)
        ax.set_ylim(-25.0, 100.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        plot.legend(prop={'size': 16}, loc="upper right",
                    bbox_to_anchor=(1.175, 1.00))
        fig.subplots_adjust(bottom=0.16, right=0.86)
        fig.savefig("%s/%s.png" % (results_dir, f_name))


if __name__ == "__main__":
    main()
