import os
import random
from multiprocessing import Queue, Process
import sys

import numpy as np
import torch
import pandas as pd

from policy_learning.deep_gmm import train_policy_deepgmm
from policy_learning.efficient_gmm_baselines import train_policy_gmm_benchmark, \
    RandomKitchenSinkGenerator, calc_norm_matrix_efficient, \
    PolynomialWeightsGenerator
from nuisance.nuisance_generator import StandardNuisanceGenerator
from policy_learning.policy_networks import LinearPolicyNetwork, \
    FlexiblePolicyNetwork
from scenarios.jobs_scenario import JobsScenario
from policy_learning.unweighted_baselines import train_policy_unweighted, \
    doubly_robust_psi


def main():
    general_experiment_arguments = {
        "num_rep": 64,
        "num_procs": 1,
        "num_gpu": 1,
        "run_finite_gmm": False,
    }

    # make results directory if necessary
    results_dir = "jobs_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # run experiment with LinearPolicy
    run_experiment(results_dir=results_dir,
                   out_file_prefix="linear_linear",
                   policy_network_class=LinearPolicyNetwork,
                   esprm_lr=0.001, esprm_epoch_data_mul=8000000,
                   esprm_max_epoch=8000, **general_experiment_arguments)

    # run experiment with FlexiblePolicy
    run_experiment(results_dir=results_dir,
                   out_file_prefix="linear_flexible",
                   policy_network_class=FlexiblePolicyNetwork,
                   esprm_lr=0.0002, esprm_epoch_data_mul=8000000,
                   esprm_max_epoch=8000, **general_experiment_arguments)



def run_experiment(results_dir, out_file_prefix, policy_network_class,
                   num_rep, num_procs, num_gpu,
                   esprm_lr, esprm_epoch_data_mul, esprm_max_epoch,
                   run_finite_gmm=False):
    batch_size = 1024
    max_num_epochs = 500
    max_no_improve = 5
    psi_function = doubly_robust_psi

    job_queue = Queue()
    results_queue = Queue()
    num_jobs = 0

    for rep in range(num_rep):
        batch_job = {
            "job_list": [],
            "seed": random.randint(0, 2 ** 32 - 1),
            "batch_size": batch_size,
            "max_num_epochs": max_num_epochs,
            "max_no_improve": max_no_improve,
            "rep": rep,
            "psi_function": psi_function,
            "policy_network_class": policy_network_class,
            "nuisance_generator_class": StandardNuisanceGenerator,
            "nuisance_generator_args": {
                "y_method": "torch",
                "p_method": "torch",
                "y_args": {},
                "p_args": {},
            },
        }

        # # unweighted nuisance job
        job = {"method": "unweighted"}
        batch_job["job_list"].append(job)
        num_jobs += 1

        # deep gmm job
        job = {"method": "deepgmm", "policy_lr": esprm_lr,
               "epoch_data_mul": esprm_epoch_data_mul,
               "deepgmm_max_num_epoch": esprm_max_epoch}
        batch_job["job_list"].append(job)
        num_jobs += 1

        if run_finite_gmm:
            # Polynomial kernel weights
            for poly_deg in (2, 3):
                job = {"method": "gmm",
                       "weights_generator_class": PolynomialWeightsGenerator,
                       "weights_generator_args": {"degree": poly_deg},
                       "norm_matrix_function": calc_norm_matrix_efficient}
                batch_job["job_list"].append(job)
                num_jobs += 1

            # RBF kernel weights
            for num_moments in (16, 32, 64):
                job = {"method": "gmm",
                       "weights_generator_class": RandomKitchenSinkGenerator,
                       "weights_generator_args": {"num_moments": num_moments},
                       "norm_matrix_function": calc_norm_matrix_efficient}
                batch_job["job_list"].append(job)
                num_jobs += 1

        job_queue.put(batch_job)

    procs = []
    for p_i in range(num_procs):
        device_i = p_i % num_gpu
        job_queue.put("STOP")
        p = Process(target=worker_function,
                    args=(device_i, job_queue, results_queue))
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


def worker_function(device_i, job_queue, results_queue):
    if torch.cuda.is_available():
        with torch.cuda.device(device_i):
            loop_jobs(job_queue, results_queue)
    else:
        loop_jobs(job_queue, results_queue)


def loop_jobs(job_queue, results_queue):

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

        rep = batch_job["rep"]
        train_df = pd.read_csv("jobs_data/train_%d.csv" % rep)
        num_dev = len(train_df) // 5
        num_tune = len(train_df) // 5
        scenario = JobsScenario(train_df, num_dev=num_dev, num_tune=num_tune)
        batch_size = batch_job["batch_size"]
        max_num_epochs = batch_job["max_num_epochs"]
        max_no_improve = batch_job["max_no_improve"]
        psi_function = batch_job["psi_function"]
        policy_network_class = batch_job["policy_network_class"]
        nuisance_generator_class = batch_job["nuisance_generator_class"]
        nuisance_generator_args = batch_job["nuisance_generator_args"]

        print("starting next batch job (iter=%d)" % rep)

        x, a, y, = scenario.get_train()
        x_tune, a_tune, y_tune, = scenario.get_tune()
        x_dev, a_dev, y_dev, = scenario.get_dev()
        # print(len(x), len(x_tune), len(x_dev))

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
                    x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, y_dev_cf=None)
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
                    x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, y_dev_cf=None)
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
                    x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, y_dev_cf=None)
                num_epoch_code = epoch_data_mul // 1000000
                weights_str = "DeepGmm:%.4f:%d" % (policy_lr, num_epoch_code)
                job_metadata = {"method": "deepgmm",
                                "weights": weights_str}

            else:
                policy_network = None
                sys.stderr.write("Invalid method: %s" % job["method"])
                job_metadata = {"method": "Invalid",
                                "weights": "Invalid"}

            results = {"rep": rep}
            results.update(job_metadata)
            if policy_network is not None:
                policy_val = evaluate_job_policy(policy_network, rep)
                results["test_policy_val"] = policy_val
                results["theta"] = policy_network.get_policy_weights()
            else:
                results["test_policy_val"] = None
                results["theta"] = None
            results_queue.put(results)

        print("finished job batch (iter=%d)" % rep)


def evaluate_job_policy(policy_network, rep):
    test_df = pd.read_csv("jobs_data/test_%d.csv" % rep)
    scenario = JobsScenario(test_df, num_dev=0, num_tune=0)
    x_test, a_test, y_test = scenario.get_all_data_for_testing()
    ipw_test = scenario.get_ipw()
    pred_a = (policy_network(x_test).view(-1) >= 0).long()
    a_match = (pred_a == a_test).double()
    policy_val = (y_test.view(-1) * ipw_test * a_match).mean()
    return float(policy_val.detach().cpu().numpy())


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
