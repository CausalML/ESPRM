import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from nuisance.r_glm import r_linear_regression

from util.input_embedding import to_quadratic
from util.tensor_util import tensor_invalid


class SimpleMeanYNetwork(nn.Module):
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
        return h

    def no_nan(self):
        for layer in (self.linear_1, self.linear_2, self.linear_3):
            if tensor_invalid(layer.weight):
                return False
            if tensor_invalid(layer.bias):
                return False
        return True


class TinyMeanYNetwork(nn.Module):
    def __init__(self, dim_x, quadratic=False):
        nn.Module.__init__(self)
        if quadratic:
            input_dim = 2 * dim_x + dim_x * (dim_x - 1) // 2
        else:
            input_dim = dim_x
        self.dim_x = dim_x
        self.input_dim = input_dim

        self.linear_1 = nn.Linear(input_dim, 50)
        self.linear_2 = nn.Linear(50, 2)
        self.quadratic = quadratic

    def forward(self, x):
        if self.quadratic:
            x = to_quadratic(x)
        h = self.linear_1(x)
        h = self.linear_2(F.leaky_relu(h))
        return h

    def no_nan(self):
        for layer in (self.linear_1, self.linear_2):
            if tensor_invalid(layer.weight):
                return False
            if tensor_invalid(layer.bias):
                return False
        return True


class RMeanYNetwork(object):
    def __init__(self, lin_reg_0, lin_reg_1, quadratic=False):
        self.lin_reg_0 = lin_reg_0
        self.lin_reg_1 = lin_reg_1
        self.quadratic = quadratic

    def __call__(self, x):
        if self.quadratic:
            x = to_quadratic(x)
        out_0 = self.lin_reg_0(x)
        out_1 = self.lin_reg_1(x)
        return torch.cat([out_0, out_1], dim=1)


def train_mean_y_network(x, a, y, method, x_dev, a_dev, y_dev, **y_args):
    if method == "torch":
        while True:
            mean_y_network = train_mean_y_network_torch(
                x, a, y, x_dev=x_dev, a_dev=a_dev, y_dev=y_dev, **y_args)
            if mean_y_network.no_nan():
                break
        return mean_y_network
    elif method == "glm":
        return train_mean_y_network_glm(x, a, y, **y_args)
    else:
        raise ValueError("Invalid method for training y network: %s" % method)


def train_mean_y_network_glm(x, a, y, quadratic=False):
    num_data = x.shape[0]
    idx_0 = [i for i in range(num_data) if a[i] == 0]
    idx_1 = [i for i in range(num_data) if a[i] == 1]
    if quadratic:
        x = to_quadratic(x)

    # fit linear regression models
    lin_reg_0 = r_linear_regression(x[idx_0], y[idx_0])
    lin_reg_1 = r_linear_regression(x[idx_1], y[idx_1])
    return RMeanYNetwork(lin_reg_0, lin_reg_1, quadratic=quadratic)


def train_mean_y_network_torch(x, a, y, batch_size=1024, num_epoch=5000,
                               verbose=False, x_dev=None, a_dev=None,
                               y_dev=None, model_type="simple",
                               max_no_improvement=5, quadratic=False,
                               do_lbfgs=True):
    num_data = x.shape[0]
    dim_x = x.shape[1]
    num_batches = num_data // batch_size + int(num_data % batch_size > 0)

    loss_function = nn.MSELoss()
    if model_type == "simple":
        mean_y_network = TinyMeanYNetwork(
            dim_x=dim_x, quadratic=quadratic).double()
    elif model_type == "flexible":
        mean_y_network = SimpleMeanYNetwork(
            dim_x=dim_x, quadratic=quadratic).double()
    else:
        raise ValueError("Invalid model type for mean y network: %s"
                         % model_type)
    if torch.cuda.is_available():
        mean_y_network.cuda()

    if do_lbfgs:
        # do initial optimization with LBFGS
        lbfgs = torch.optim.LBFGS(mean_y_network.parameters())

        def closure():
            lbfgs.zero_grad()
            pred_y_ = torch.gather(mean_y_network(x), 1, a.view(-1, 1))
            loss_ = loss_function(pred_y_, y)
            loss_.backward()
            return loss_
        lbfgs.step(closure)

    optimizer = torch.optim.Adam(mean_y_network.parameters())

    min_dev_loss = float("inf")
    no_improvement = 0

    for epoch in range(num_epoch):
        idx = list(range(num_data))
        random.shuffle(idx)
        idx_cycle = itertools.cycle(idx)

        if verbose and epoch % 10 == 0:
            print("starting epoch", epoch + 1)
            pred_y_dev = torch.gather(mean_y_network(x_dev), 1,
                                      a_dev.view(-1, 1))
            dev_loss = loss_function(pred_y_dev, y_dev)
            print("dev loss:", float(dev_loss))
            # print(pred_y_dev[:10].detach().numpy(), y_dev[:10].detach().numpy())

        for i in range(num_batches):
            batch_idx = [next(idx_cycle) for _ in range(batch_size)]
            x_batch = x[batch_idx]
            a_batch = a[batch_idx]
            y_batch = y[batch_idx]
            pred_y_batch = torch.gather(mean_y_network(x_batch), 1,
                                        a_batch.view(-1, 1))
            loss = loss_function(pred_y_batch, y_batch)
            # loss = ((pred_y_batch - y_batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if max_no_improvement:
            pred_y_dev = torch.gather(mean_y_network(x_dev), 1,
                                      a_dev.view(-1, 1))
            dev_loss = loss_function(pred_y_dev, y_dev)
            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > max_no_improvement:
                break

    if verbose:
        pred_y_dev = torch.gather(mean_y_network(x_dev), 1,
                                  a_dev.view(-1, 1))
        dev_loss = loss_function(pred_y_dev, y_dev)
        print("final dev loss:", float(dev_loss))
    return mean_y_network
