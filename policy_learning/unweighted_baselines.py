import random
import itertools

import torch


def logistic_loss_function(psi, policy_out):
    sign_psi = ((psi > 0).long() * 2 - 1).double()
    # loss = (psi.abs() * torch.log(torch.sigmoid(-sign_psi * policy_out))).mean()
    loss = (psi.abs() * (2 * torch.log(1 + torch.exp(policy_out))
                         - (sign_psi + 1) * policy_out)).mean()
    return loss


def mean_outcome_psi(x, a, y, propensity_network, mean_y_network):
    mean_y_pred = mean_y_network(x)
    psi = (mean_y_pred[:, 1] - mean_y_pred[:, 0]).reshape(-1, 1)
    return psi


def ipw_psi(x, a, y, propensity_network, mean_y_network):
    q = torch.gather(propensity_network(x), 1, a.view(-1, 1))
    sign_a = (2 * a - 1).view(-1, 1).double()
    psi = sign_a * y / q
    return psi


def doubly_robust_psi(x, a, y, propensity_network, mean_y_network):
    mean_y_pred = mean_y_network(x)
    mean_y_pred_a = torch.gather(mean_y_pred, 1, a.view(-1, 1))
    q = torch.gather(propensity_network(x), 1, a.view(-1, 1))
    psi_mean_outcome = (mean_y_pred[:, 1] - mean_y_pred[:, 0]).reshape(-1, 1)
    sign_a = (2 * a - 1).view(-1, 1).double()
    psi = psi_mean_outcome + sign_a * (y - mean_y_pred_a) / q
    return psi


def train_policy_unweighted(x, a, y, batch_size, max_num_epoch, psi_function,
                            policy_network_class, nuisance_generator,
                            verbose=False, max_no_improve=None, x_dev=None,
                            a_dev=None, y_dev=None, y_dev_cf=None,
                            do_lbfgs=True):
    num_data = x.shape[0]
    dim_x = x.shape[1]
    num_batches = num_data // batch_size + int(num_data % batch_size > 0)

    policy_network = policy_network_class(dim_x=dim_x).double()
    if torch.cuda.is_available():
        policy_network.cuda()
    # policy_network = FlexiblePolicyNetwork(dim_x=dim_x).double()
    mean_y_network = nuisance_generator.get_mean_y_network()
    propensity_network = nuisance_generator.get_propensity_network()

    # do initial optimization with LBFGS
    if do_lbfgs:
        lbfgs = torch.optim.LBFGS(policy_network.parameters())

        def closure():
            lbfgs.zero_grad()
            psi_ = psi_function(x, a, y, propensity_network, mean_y_network)
            policy_out_ = policy_network(x)
            loss_ = logistic_loss_function(psi_, policy_out_)
            loss_.backward()
            return loss_
        lbfgs.step(closure)

    optimizer = torch.optim.Adam(policy_network.parameters(),
                                 weight_decay=0.00)
    min_dev_loss = float("inf")
    no_improve = 0

    for epoch in range(max_num_epoch):
        if max_no_improve:
            policy_out_dev = policy_network(x_dev).detach()
            psi_dev = psi_function(x_dev, a_dev, y_dev, propensity_network,
                                   mean_y_network)
            dev_loss = logistic_loss_function(psi_dev, policy_out_dev)

            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    # print("broke after %d epochs" % epoch)
                    break

        if verbose and epoch % 100 == 0:
            policy_out_dev = policy_network(x_dev).detach()
            policy_a = (policy_out_dev > 0).long()
            y_dev_selected = torch.gather(y_dev_cf, 1, policy_a)
            dev_policy_val = float(y_dev_selected.mean().cpu())
            # print(policy_out_dev[:10])
            print("epoch %d, dev policy val %.4f"
                  % (epoch, dev_policy_val))

        idx = list(range(num_data))
        random.shuffle(idx)
        idx_cycle = itertools.cycle(idx)
        for i in range(num_batches):
            batch_idx = [next(idx_cycle) for _ in range(batch_size)]
            x_batch = x[batch_idx]
            a_batch = a[batch_idx]
            y_batch = y[batch_idx]

            psi = psi_function(x_batch, a_batch, y_batch,
                               propensity_network, mean_y_network)
            policy_out = policy_network(x_batch)
            loss = logistic_loss_function(psi, policy_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if verbose:
        policy_out_dev = policy_network(x_dev).detach()
        policy_a = (policy_out_dev > 0).long()
        y_dev_selected = torch.gather(y_dev_cf, 1, policy_a)
        dev_policy_val = float(y_dev_selected.mean().cpu())
        # print(x_dev[:10])
        # print(policy_out_dev[:10])
        print("final dev policy val: %.4f:" % dev_policy_val)

    return policy_network

