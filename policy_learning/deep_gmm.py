import random
import itertools

import torch.nn as nn
import torch.nn.functional as F
import torch

from policy_learning.unweighted_baselines import train_policy_unweighted
from util.oadam import OAdam


class FlexibleCriticNetwork(nn.Module):
    def __init__(self, dim_x):
        nn.Module.__init__(self)
        self.dim_x = dim_x
        self.linear_1 = nn.Linear(self.dim_x, 50)
        self.linear_2 = nn.Linear(50, 1)

    def forward(self, x):
        h = self.linear_1(x)
        return self.linear_2(F.leaky_relu(h))


def compute_game_objectives(critic_out, policy_out, psi):
    sign_psi = ((psi > 0).long() * 2 - 1).double()
    l_prime = 2 * torch.sigmoid(policy_out) - (sign_psi + 1)
    phi_mean = (critic_out * psi.abs() * l_prime).mean()
    phi_squared_mean = ((critic_out * psi.abs() * l_prime) ** 2).mean()
    return phi_mean, -phi_mean + 0.25 * phi_squared_mean,


def compute_dev_loss(policy_out_dev, critic_out_dev_list, psi_dev):
    sign_psi = ((psi_dev > 0).long() * 2 - 1).double()
    l_prime = 2 * torch.sigmoid(policy_out_dev) - (sign_psi + 1)
    phi_mean_list = []
    for critic_out_dev in critic_out_dev_list:
        phi_mean = (critic_out_dev * psi_dev.abs() * l_prime).mean()
        phi_squared_mean = ((critic_out_dev * psi_dev.abs()
                             * l_prime) ** 2).mean()
        phi_mean_list.append(phi_mean - 0.25 * phi_squared_mean)
    return max(phi_mean_list)



def train_policy_deepgmm(x, a, y, batch_size, psi_function,
                         policy_network_class, nuisance_generator,
                         initialize_from_erm=False,
                         max_num_epoch=4000, epoch_data_mul=4000000,
                         verbose=False, policy_lr=0.005, lbfgs_freq=10,
                         print_freq=10, x_dev=None, a_dev=None,
                         y_dev=None, y_dev_cf=None):
    num_data = x.shape[0]
    dim_x = x.shape[1]
    num_batches = num_data // batch_size + int(num_data % batch_size > 0)
    num_epoch = min(max_num_epoch, epoch_data_mul // num_data)

    # init policy_network using unweighted method
    if initialize_from_erm:
        policy_network = train_policy_unweighted(
            x=x, a=a, y=y, batch_size=batch_size, max_num_epoch=5000,
            psi_function=psi_function,
            policy_network_class=policy_network_class,
            nuisance_generator=nuisance_generator, max_no_improve=5,
            x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, y_dev_cf=y_dev_cf,
            verbose=verbose)
    else:
        policy_network = policy_network_class(dim_x=dim_x).double()
    if torch.cuda.is_available():
        policy_network.cuda()
    mean_y_network = nuisance_generator.get_mean_y_network()
    propensity_network = nuisance_generator.get_propensity_network()
    critic_network = FlexibleCriticNetwork(dim_x=dim_x).double()
    if torch.cuda.is_available():
        critic_network = critic_network.cuda()
    critic_lr = 5 * policy_lr
    policy_optimizer = OAdam(policy_network.parameters(),
                             lr=policy_lr, betas=(0.5, 0.9), weight_decay=0.00)
    critic_optimizer = OAdam(critic_network.parameters(),
                             lr=critic_lr, betas=(0.5, 0.9), weight_decay=0.00)

    if verbose:
        psi_dev = psi_function(x_dev, a_dev, y_dev,
                               propensity_network, mean_y_network)
        psi_dev = psi_dev.detach().cpu()
        y_dev_cf = y_dev_cf.cpu()
        critic_out_dev_list = []

    for epoch in range(num_epoch):
        if verbose and epoch % 5 == 0 and epoch < (num_epoch // 2):
            critic_out_dev = critic_network(x_dev).detach().cpu()
            critic_out_dev_list.append(critic_out_dev)
        if verbose and epoch % print_freq == 0:
            policy_out_dev = policy_network(x_dev).detach().cpu()
            policy_a = (policy_out_dev > 0).long()
            y_dev_selected = torch.gather(y_dev_cf, 1, policy_a)
            dev_policy_val = float(y_dev_selected.mean())
            policy_loss_dev = compute_dev_loss(
                policy_out_dev, critic_out_dev_list, psi_dev)
            print("epoch %d, dev policy val %.4f, current obj %.4f"
                  % (epoch, dev_policy_val, float(policy_loss_dev)))

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
            critic_out = critic_network(x_batch)
            policy_loss, critic_loss = compute_game_objectives(
                critic_out, policy_out, psi)

            critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

    if verbose:
        policy_out_dev = policy_network(x_dev).detach()
        policy_a = (policy_out_dev > 0).long()
        y_dev_selected = torch.gather(y_dev_cf, 1, policy_a)
        dev_policy_val = float(y_dev_selected.mean().cpu())
        print("final dev policy val: %.4f:" % dev_policy_val)

    return policy_network


