# ESPRM

Code for paper <i>Efficient Policy Learning from Surrogate-Loss Classification
Reductions</i>, Bennett, Andrew and Kallus, Nathan
(https://arxiv.org/abs/2002.05153).

Can run synethetic experiments by running script
"run_synthetic_experiments.py", and experiments on jobs case study by running
script "run_jobs_experiments.py". Note that in order to run the jobs case
study experiments first the data needs to be converted into the required format
by running the script "make_jobs_data.r". Various aspects of the experiments
can be customized by editing the "main" functions and the start of the
"run_experiments" functions in the respective scripts. Of particular
note the variable "num_procs" can be changed to a value greater
than 1 to allow multi-processing, and "num_gpu" can be changed to a value
greater than to use multiple GPUs if multiple are available
(the variable has no effect if no GPUs are available). Note that the code will
automatically make use of GPU if it is available on the machine.

Then graphs can be created from the results using the "make_*_graphs.py"
scripts, and the results of the jobs experiment can be analyzed using the
"analyze_job_results.py" script.

Any questions should be directed to awb222@cornell.edu.
