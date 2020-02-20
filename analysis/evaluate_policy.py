import torch


def evaluate_policy_cf_data(policy_network, x_test, y_test_cf):
    policy_out_dev = policy_network(x_test).detach()
    policy_a = (policy_out_dev > 0).long()
    y_dev_selected = torch.gather(y_test_cf, 1, policy_a)
    dev_policy_val = float(y_dev_selected.mean())
    return dev_policy_val

