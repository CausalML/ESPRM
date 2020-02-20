import pandas as pd
import numpy as np
from scipy.stats import ttest_rel


file_names = ("jobs_linear", "jobs_flexible")
results_dir = "final_results"
num_reps = 64

def main():
    for f_name in file_names:
        path = "%s/%s.csv" % (results_dir, f_name)
        df = pd.read_csv(path, index_col=0)

        our_results = np.zeros(num_reps)
        bench_results = np.zeros(num_reps)
        for i in range(len(df)):
            method = df["weights"][i]
            val = df["test_policy_val"][i]
            rep = int(df["rep"][i])
            if "Deep" in method:
                our_results[rep] = val
            else:
                bench_results[rep] = val

        diffs = our_results - bench_results
        mean_diff = diffs.mean()
        ci = diffs.std() * 1.96 / (num_reps ** 0.5)
        ci_mul = 1.96 / (num_reps ** 0.5)

        print("ERM: %.5f ± %.5f" % (float(bench_results.mean()),
                                    float(bench_results.std() * ci_mul)))
        print("Ours: %.5f ± %.5f" % (float(our_results.mean()),
                                     float(our_results.std() * ci_mul)))
        print(f_name, "mean_diff %.5f ± %.5f" % (float(mean_diff),
                                                 float(diffs.std())),
              "confidence interval %.5f" % ci)
        print(ttest_rel(our_results, bench_results))


if __name__ == "__main__":
    main()
