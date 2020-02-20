import random
from multiprocessing import Queue, Process
import sys
import os

import numpy as np
import torch
import pandas as pd

from analysis.evaluate_policy import evaluate_policy_cf_data
from analysis.scenario_statistics import get_scenario_statistics
from policy_learning.deep_gmm import train_policy_deepgmm
from policy_learning.unweighted_baselines import train_policy_unweighted, \
    doubly_robust_psi
from policy_learning.efficient_gmm_baselines import train_policy_gmm_benchmark,\
    RandomKitchenSinkGenerator, calc_norm_matrix_efficient, \
    PolynomialWeightsGenerator
from nuisance.nuisance_generator import StandardNuisanceGenerator
from policy_learning.policy_networks import LinearPolicyNetwork, \
    FlexiblePolicyNetwork
from scenarios.simple_scenario import RandomSimpleScenario
from scenarios.quadratic_scenario import RandomQuadraticScenario


def main():
    general_experiment_arguments = {
        "num_rep": 64,
        "num_procs": 1,
        "num_gpu": 1,
        "flexible_nuisance": False,
        "run_finite_gmm": False,
    }

    # make results directory if necessary
    results_dir = "synthetic_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # run experiment with LinearPolicy, LinearScenario
    run_experiment(results_dir=results_dir,
                   out_file_prefix="linear_linear",
                   scenario_class=RandomSimpleScenario,
                   policy_network_class=LinearPolicyNetwork,
                   esprm_lr=0.001, esprm_epoch_data_mul=8000000,
                   esprm_max_epoch=8000, quadratic=False,
                   **general_experiment_arguments)

    # run experiment with FlexiblePolicy, LinearScenario
    run_experiment(results_dir=results_dir,
                   out_file_prefix="linear_flexible",
                   scenario_class=RandomSimpleScenario,
                   policy_network_class=FlexiblePolicyNetwork,
                   esprm_lr=0.0002, esprm_epoch_data_mul=8000000,
                   esprm_max_epoch=8000, quadratic=False,
                   **general_experiment_arguments)

    # run experiment with LinearPolicy, QuadraticScenario
    run_experiment(results_dir=results_dir,
                   out_file_prefix="quadratic_linear",
                   scenario_class=RandomQuadraticScenario,
                   policy_network_class=LinearPolicyNetwork,
                   esprm_lr=0.001, esprm_epoch_data_mul=8000000,
                   esprm_max_epoch=8000, quadratic=True,
                   **general_experiment_arguments)

    # run experiment with FlexiblePolicy, QuadraticScenario
    run_experiment(results_dir=results_dir,
                   out_file_prefix="quadratic_flexible",
                   scenario_class=RandomQuadraticScenario,
                   policy_network_class=FlexiblePolicyNetwork,
                   esprm_lr=0.0002, esprm_epoch_data_mul=8000000,
                   esprm_max_epoch=8000, quadratic=True,
                   **general_experiment_arguments)


def run_experiment(results_dir, out_file_prefix, policy_network_class,
                   scenario_class, num_rep, num_procs, num_gpu,
                   esprm_lr, esprm_epoch_data_mul, esprm_max_epoch,
                   flexible_nuisance=False, quadratic=False,
                   run_finite_gmm=False):
    num_test = 1000000
    batch_size = 1024
    max_num_epochs = 500
    max_no_improve = 5
    num_train_range = (10000, 5000, 2000, 1000, 500, 200, 100)
    psi_function = doubly_robust_psi
    if flexible_nuisance:
        nuisance_method = "torch"
        nuisance_args = {}
    else:
        nuisance_method = "glm"
        nuisance_args = {"quadratic": quadratic}

    job_queue = Queue()
    results_queue = Queue()
    num_jobs = 0

    for num_train in num_train_range:
        for rep in range(num_rep):
            batch_job = {
                "job_list": [],
                "seed": random.randint(0, 2 ** 32 - 1),
                "batch_size": batch_size,
                "num_train": num_train,
                "num_tune": num_train,
                "num_dev": num_train,
                "num_test": num_test,
                "max_num_epochs": max_num_epochs,
                "max_no_improve": max_no_improve,
                "rep": rep,
                "psi_function": psi_function,
                "policy_network_class": policy_network_class,
                "job_scenario_args": {},
                "nuisance_generator_class": StandardNuisanceGenerator,
                "nuisance_generator_args": {
                    "y_method": nuisance_method,
                    "p_method": nuisance_method,
                    "y_args": nuisance_args,
                    "p_args": nuisance_args,
                },
            }

            # ERM job
            job = {"method": "unweighted"}
            batch_job["job_list"].append(job)
            num_jobs += 1

            # ESPRM job
            job = {"method": "deepgmm", "policy_lr": esprm_lr,
                   "epoch_data_mul": esprm_epoch_data_mul,
                   "deepgmm_max_num_epoch": esprm_max_epoch}
            batch_job["job_list"].append(job)
            num_jobs += 1

            # FiniteGMM jobs
            if run_finite_gmm:
                # Polynomial kernel weights
                for poly_deg in (2, 3):
                    job = {
                        "method": "gmm",
                        "weights_generator_class": PolynomialWeightsGenerator,
                        "weights_generator_args": {"degree": poly_deg},
                        "norm_matrix_function": calc_norm_matrix_efficient
                    }
                    batch_job["job_list"].append(job)
                    num_jobs += 1

                # RBF kernel weights
                for num_moments in (16, 32, 64):
                    job = {
                        "method": "gmm",
                        "weights_generator_class": RandomKitchenSinkGenerator,
                        "weights_generator_args": {"num_moments": num_moments},
                        "norm_matrix_function": calc_norm_matrix_efficient
                    }
                    batch_job["job_list"].append(job)
                    num_jobs += 1

            job_queue.put(batch_job)

    procs = []
    for p_i in range(num_procs):
        device_i = p_i % num_gpu
        job_queue.put("STOP")
        p = Process(target=worker_function,
                    args=(device_i, scenario_class, job_queue, results_queue))
        p.start()
        procs.append(p)

    results_list = []
    for _ in range(num_jobs):
        results = results_queue.get()
        results_list.append(results)

    for p in procs:
        p.join()
    out_df = results_list_to_data_frame(results_list)
    out_df.to_csv("%s/%s.csv" % (results_dir, out_file_prefix))


def worker_function(device_i, scenario_class, job_queue, results_queue):
    if torch.cuda.is_available():
        with torch.cuda.device(device_i):
            loop_jobs(scenario_class, job_queue, results_queue)
    else:
        loop_jobs(scenario_class, job_queue, results_queue)


def loop_jobs(scenario_class, job_queue, results_queue):

    for batch_job in iter(job_queue.get, "STOP"):

        # set random seed
        starting_seed = batch_job["seed"]
        random.seed(starting_seed)
        try:
            np.random.seed(starting_seed)
        except:
            print(starting_seed)
            np.random.seed(starting_seed % (2 ** 32))
            print("seed fail")
        torch.manual_seed(starting_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(starting_seed)

        scenario = scenario_class()
        batch_size = batch_job["batch_size"]
        num_train = batch_job["num_train"]
        num_tune = batch_job["num_tune"]
        num_dev = batch_job["num_dev"]
        num_test = batch_job["num_test"]
        max_num_epochs = batch_job["max_num_epochs"]
        max_no_improve = batch_job["max_no_improve"]
        rep = batch_job["rep"]
        psi_function = batch_job["psi_function"]
        job_scenario_args = batch_job["job_scenario_args"]
        policy_network_class = batch_job["policy_network_class"]
        nuisance_generator_class = batch_job["nuisance_generator_class"]
        nuisance_generator_args = batch_job["nuisance_generator_args"]

        print("starting next batch job (num_train=%d, rep=%d)"
              % (num_train, rep))

        scenario.initialize(**job_scenario_args)
        x, a, y, _ = scenario.sample_data(num_train)
        x_tune, a_tune, y_tune, _ = scenario.sample_data(num_tune)
        x_dev, a_dev, y_dev, y_dev_cf = scenario.sample_data(num_dev)
        x_test, _, _, y_test_cf = scenario.sample_data(num_test)

        # obtain general results
        scenario_statistics = get_scenario_statistics(
            scenario, FlexiblePolicyNetwork, x_test, y_test_cf)
        theta_dict = scenario.get_theta_dict()
        batch_statistics = {
            "num_train": num_train,
            "starting_seed": starting_seed,
            "rep": rep,
        }
        batch_statistics.update(scenario_statistics)
        batch_statistics.update(theta_dict)

        nuisance_generator = nuisance_generator_class(
            scenario, **nuisance_generator_args)
        nuisance_generator.setup(x_tune, a_tune, y_tune, x_dev, a_dev, y_dev)

        for job in batch_job["job_list"]:

            if job["method"] == "unweighted":
                policy_network = train_policy_unweighted(
                    x=x, a=a, y=y, batch_size=batch_size,
                    max_num_epoch=max_num_epochs,
                    max_no_improve=max_no_improve,
                    psi_function=psi_function,
                    nuisance_generator=nuisance_generator,
                    policy_network_class=policy_network_class, verbose=False,
                    x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, y_dev_cf=y_dev_cf)
                job_metadata = {"method": "unweighted",
                                "weights": "Unweighted"}

            elif job["method"] == "gmm":
                weights_class = job["weights_generator_class"]
                weights_args = job["weights_generator_args"]
                norm_matrix_function = job["norm_matrix_function"]
                weights_generator = weights_class(
                    x_tune, a_tune, y_tune, x_dev, a_dev, y_dev, **weights_args)
                policy_network = train_policy_gmm_benchmark(
                    x=x, a=a, y=y, batch_size=batch_size,
                    num_stages=3, max_num_epoch_per_stage=max_num_epochs,
                    max_no_improve=max_no_improve, psi_function=psi_function,
                    nuisance_generator=nuisance_generator,
                    policy_network_class=policy_network_class, verbose=False,
                    weights_function=weights_generator,
                    norm_matrix_function=norm_matrix_function,
                    x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, y_dev_cf=y_dev_cf)
                job_metadata = {"method": "gmm",
                                "weights": str(weights_generator)}

            elif job["method"] == "deepgmm":
                policy_lr = job["policy_lr"]
                epoch_data_mul = job["epoch_data_mul"]
                deepgmm_max_num_epoch = job["deepgmm_max_num_epoch"]
                policy_network = train_policy_deepgmm(
                    x=x, a=a, y=y, batch_size=batch_size,
                    psi_function=psi_function, policy_lr=policy_lr,
                    epoch_data_mul=epoch_data_mul,
                    max_num_epoch=deepgmm_max_num_epoch,
                    nuisance_generator=nuisance_generator,
                    policy_network_class=policy_network_class, verbose=False,
                    x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, y_dev_cf=y_dev_cf)
                num_epoch_code = epoch_data_mul // 1000000
                weights_str = "DeepGmm:%.4f:%d" % (policy_lr, num_epoch_code)
                job_metadata = {"method": "deepgmm",
                                "weights": weights_str}

            else:
                policy_network = None
                sys.stderr.write("Invalid method: %s" % job["method"])
                job_metadata = {"method": "Invalid",
                                "weights": "Invalid"}

            results = {}
            results.update(batch_statistics)
            results.update(job_metadata)
            if policy_network is not None:
                results["test_policy_val"] = evaluate_policy_cf_data(
                    policy_network, x_test, y_test_cf)
                results["theta"] = policy_network.get_policy_weights()
            else:
                results["test_policy_val"] = None
                results["theta"] = None
            results_queue.put(results)

        print("finished job batch (num_train=%d, rep=%d)" % (num_train, rep))


def results_list_to_data_frame(results_list):
    keys = {k for results in results_list for k in results.keys()}
    data_frame_dict = {}
    for k in keys:
        vals = [results[k] if k in results else None
                for results in results_list]
        data_frame_dict[k] = np.array(vals)
    return pd.DataFrame(data_frame_dict)



if __name__ == "__main__":
    main()
