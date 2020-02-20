import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from nuisance.r_glm import r_logistic_regression
from util.input_embedding import to_quadratic
from util.tensor_util import tensor_invalid

class SimplePropensityNetwork(nn.Module):
    def __init__(self, dim_x, quadratic=False):
        nn.Module.__init__(self)
        if quadratic:
            input_dim = 2 * dim_x + dim_x * (dim_x - 1) // 2
        else:
            input_dim = dim_x
        self.dim_x = dim_x
        self.input_dim = input_dim

        self.linear_1 = nn.Linear(self.input_dim, 100)
        self.linear_2 = nn.Linear(100, 100)
        self.linear_3 = nn.Linear(100, 2)
        self.quadratic = quadratic

    def forward(self, x):
        if self.quadratic:
            x = to_quadratic(x)
        h = self.linear_1(x)
        h = self.linear_2(F.leaky_relu(h))
        h = self.linear_3(F.leaky_relu(h))
        return F.softmax(h, dim=1)

    def output_final_hidden(self, x):
        if self.quadratic:
            x = to_quadratic(x)
        h = self.linear_1(x)
        return self.linear_2(F.leaky_relu(h))

    def no_nan(self):
        for layer in (self.linear_1, self.linear_2, self.linear_3):
            if tensor_invalid(layer.weight):
                return False
            if tensor_invalid(layer.bias):
                return False
        return True


class TinyPropensityNetwork(nn.Module):
    def __init__(self, dim_x, quadratic=False):
        nn.Module.__init__(self)
        if quadratic:
            input_dim = 2 * dim_x + dim_x * (dim_x - 1) // 2
        else:
            input_dim = dim_x
        self.dim_x = dim_x
        self.input_dim = input_dim

        self.linear_1 = nn.Linear(self.input_dim, 50)
        self.linear_2 = nn.Linear(50, 2)
        self.quadratic = quadratic

    def forward(self, x):
        if self.quadratic:
            x = to_quadratic(x)
        h = self.linear_1(x)
        h = self.linear_2(F.leaky_relu(h))
        return F.softmax(h, dim=1)

    def output_final_hidden(self, x):
        if self.quadratic:
            x = to_quadratic(x)
        return self.linear_1(x)

    def no_nan(self):
        for layer in (self.linear_1, self.linear_2):
            if tensor_invalid(layer.weight):
                return False
            if tensor_invalid(layer.bias):
                return False
        return True


class RPropensityNetwork(object):
    def __init__(self, log_reg, quadratic=False):
        self.log_reg = log_reg
        self.quadratic = quadratic

    def __call__(self, x):
        if self.quadratic:
            x = to_quadratic(x)
        out_1 = self.log_reg(x)
        out_0 = 1.0 - out_1
        return torch.cat([out_0, out_1], dim=1)


class RTorchPropensityNetwork(object):
    def __init__(self, log_reg, torch_network):
        self.log_reg = log_reg
        self.torch_network = torch_network

    def __call__(self, x):
        x_transform = self.torch_network(x)[:, 0].view(-1, 1)
        # x_transform = self.torch_network.output_final_hidden(x)
        out_1 = self.log_reg(x_transform)
        out_0 = 1.0 - out_1
        return torch.cat([out_0, out_1], dim=1)


def train_propensity_network(x, a, method, x_dev, a_dev, **p_args):
    if method == "torch":
        while True:
            propensity_network = train_propensity_network_torch(
                x, a, x_dev=x_dev, a_dev=a_dev, **p_args)
            if propensity_network.no_nan():
                break
        return propensity_network
    elif method == "glm":
        return train_propensity_network_glm(x, a, **p_args)
    elif method == "torchglm":
        return train_propensity_network_torch_glm(
            x, a, x_dev=x_dev, a_dev=a_dev, **p_args)
    else:
        raise ValueError("Invalid method for training y network: %s" % method)


def train_propensity_network_glm(x, a, quadratic=False):
    if quadratic:
        x = to_quadratic(x)

    # fit logistic regression model
    log_reg = r_logistic_regression(x, a)
    return RPropensityNetwork(log_reg, quadratic=quadratic)


def train_propensity_network_torch_glm(x, a, x_dev, a_dev, **torch_args):
    torch_network = train_propensity_network_torch(
        x, a, x_dev=x_dev, a_dev=a_dev, **torch_args)
    x_transform = torch_network(x)[:, 0].view(-1, 1)
    # x_transform = torch_network.output_final_hidden(x)
    print(x_transform.shape)
    log_reg = r_logistic_regression(x_transform, a)
    return RTorchPropensityNetwork(log_reg, torch_network)


class PropensityNetworkFromNumpy(object):
    def __init__(self, propensity_function):
        self.propensity_function = propensity_function

    def __call__(self, x):
        x_np = x.detach().cpu().numpy()
        out = self.propensity_function(x_np)
        result = torch.from_numpy(out).double()
        if torch.cuda.is_available():
            result = result.cuda()
        return result


def get_platt_scaled_network(propensity_network, x, a, from_numpy=False):
    if from_numpy:
        propensity_network = PropensityNetworkFromNumpy(propensity_network)

    x_transform = propensity_network(x)[:, 0].view(-1, 1)
    log_reg = r_logistic_regression(x_transform, a)
    return RTorchPropensityNetwork(log_reg, propensity_network)


def logistic_loss(output, target):
    logit = torch.log(output[:, 1]) - torch.log(output[:, 0])
    target = target.double()
    # exp_logit = output[:, 1] / output[:, 0]
    # exp_logit_plus_1 = 1 / output[:, 0]
    # log_exp_logit_plus_1
    loss = (2 * -torch.log(output[:, 0]) - 2 * target * logit).mean()
    return loss


def train_propensity_network_torch(x, a, batch_size=1024, num_epoch=5000,
                                   verbose=False, x_dev=None, a_dev=None,
                                   model_type="simple",
                                   max_no_improvement=5, quadratic=False,
                                   do_lbfgs=True):
    num_data = x.shape[0]
    dim_x = x.shape[1]
    num_batches = num_data // batch_size + int(num_data % batch_size > 0)

    nll_loss = nn.NLLLoss()
    loss_function = lambda x_, t_: nll_loss(torch.log(x_), t_)
    # loss_function = logistic_loss
    if model_type == "simple":
        propensity_network = TinyPropensityNetwork(
            dim_x=dim_x, quadratic=quadratic).double()
    elif model_type == "flexible":
        propensity_network = SimplePropensityNetwork(
            dim_x=dim_x, quadratic=quadratic).double()
    else:
        raise ValueError("Invalid model type for propensity network: %s"
                         % model_type)
    if torch.cuda.is_available():
        propensity_network.cuda()

    if do_lbfgs:
        # do initial optimization with LBFGS
        lbfgs = torch.optim.LBFGS(propensity_network.parameters())

        def closure():
            lbfgs.zero_grad()
            loss_ = loss_function(propensity_network(x), a)
            loss_.backward()
            return loss_
        lbfgs.step(closure)

    optimizer = torch.optim.Adam(propensity_network.parameters())

    min_dev_loss = float("inf")
    no_improvement = 0

    for epoch in range(num_epoch):
        idx = list(range(num_data))
        random.shuffle(idx)
        idx_cycle = itertools.cycle(idx)

        if verbose and epoch % 5 == 0:
            print("starting epoch", epoch + 1)
            propensity_pred_dev = propensity_network(x_dev)
            dev_loss = loss_function(propensity_pred_dev, a_dev)
            print("dev loss:", float(dev_loss))

        for i in range(num_batches):
            batch_idx = [next(idx_cycle) for _ in range(batch_size)]
            x_batch = x[batch_idx]
            a_batch = a[batch_idx]
            loss = loss_function(propensity_network(x_batch), a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if max_no_improvement:
            propensity_pred_dev = propensity_network(x_dev)
            dev_loss = loss_function(propensity_pred_dev, a_dev)
            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > max_no_improvement:
                break

    if verbose:
        dev_loss = loss_function(propensity_network(x_dev), a_dev)
        print("final dev loss:", float(dev_loss))
    return propensity_network


