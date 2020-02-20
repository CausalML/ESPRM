import random
import itertools
import sys

import numpy as np
import scipy
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
import torch


class WeightsGenerator(object):
    def __init__(self, x_train, a_train, y_train, x_dev, a_dev, y_dev):
        self._fit(x_train, a_train, y_train, x_dev, a_dev, y_dev)

    def _fit(self, x_train, a_train, y_train, x_dev, a_dev, y_dev):
        raise NotImplementedError()

    def __call__(self, x_batch):
        raise NotImplementedError()


class PolynomialWeightsGenerator(WeightsGenerator):
    def __init__(self, x_train, a_train, y_train,
                 x_dev, a_dev, y_dev, degree=2):
        self.degree = degree
        self.sampler = PolynomialFeatures(degree)
        WeightsGenerator.__init__(self, x_train, a_train, y_train,
                                  x_dev, a_dev, y_dev)

    def _fit(self, x_train, a_train, y_train, x_dev, a_dev, y_dev):
        self.sampler.fit(x_train.detach().cpu().numpy())

    def __call__(self, x_batch):
        out = self.sampler.transform(x_batch.detach().cpu().numpy())
        out = torch.from_numpy(out).double()
        if torch.cuda.is_available():
            out = out.cuda()
        return out

    def __str__(self):
        return "poly_%d" % self.degree


class RandomKitchenSinkGenerator(WeightsGenerator):
    def __init__(self, x_train, a_train, y_train, x_dev, a_dev, y_dev,
                 num_moments, gamma=0.5):
        self.num_moments = num_moments
        self.sampler = RBFSampler(gamma=gamma, n_components=num_moments)
        WeightsGenerator.__init__(self, x_train, a_train, y_train,
                                  x_dev, a_dev, y_dev)

    def _fit(self, x_train, a_train, y_train, x_dev, a_dev, y_dev):
        self.sampler.fit(x_train.detach().cpu().numpy())

    def __call__(self, x_batch):
        out = self.sampler.transform(x_batch.detach().cpu().numpy())
        out = torch.from_numpy(out).double().view(-1, self.num_moments)
        if torch.cuda.is_available():
            out = out.cuda()
        return out

    def __str__(self):
        return "rks_%d" % self.num_moments


def calc_norm_matrix_efficient(phi, reg=0.0):
    def sym_(m_):
        return 0.5 * (m_ + m_.T)

    num_moments = phi.shape[1]
    phi = phi.detach()
    c = (phi.view(-1, num_moments, 1) * phi.view(-1, 1, num_moments))
    c = c.mean(0).cpu().numpy()
    try:
        c_sqrt = sym_(scipy.linalg.sqrtm(sym_(c)))
    except:
        sys.stderr.write("WARNING: MATRIX SQRT FAIL")
        c_sqrt = np.identity(num_moments)
    if reg > 0:
        try:
            c_inv_reg = np.linalg.inv(sym_(c) + reg * np.identity(num_moments))
        except:
            c_inv_reg = np.identity(num_moments)
            sys.stderr.write("WARNING: MATRIX INVERSE FAIL")
        c_sqrt_inv = sym_(c_inv_reg @ c_sqrt)
    else:
        try:
            c_sqrt_inv = sym_(np.linalg.inv(c_sqrt))
        except:
            c_sqrt_inv = np.identity(num_moments)
            sys.stderr.write("WARNING: MATRIX INVERSE FAIL")
    norm_matrix = torch.from_numpy(c_sqrt_inv.real).double()
    if torch.cuda.is_available():
        norm_matrix = norm_matrix.cuda()
    return norm_matrix


def calc_norm_matrix_diagonal(phi):
    phi = phi.detach()
    v = (phi ** 2).mean(0)
    return torch.diag(v ** -0.5)


def calc_norm_matrix_euclidean(phi):
    num_moments = phi.shape[1]
    return torch.eye(num_moments).double()


def compute_phi(x_weights, policy_out, psi):
    sign_psi = ((psi > 0).long() * 2 - 1).double()
    phi = x_weights * psi.abs() * (2 * torch.sigmoid(policy_out)
                                   - (sign_psi + 1))
    return phi


def gmm_benchmark_loss(x_weights, policy_out, psi, norm_matrix):
    phi = compute_phi(x_weights, policy_out, psi)
    e_moments = phi.mean(0)
    square_gmm_norm = norm_matrix.matmul(e_moments).matmul(e_moments)
    return square_gmm_norm


def evaluate_policy_dev(policy_network, x_dev, y_dev_cf):
    policy_out_dev = policy_network(x_dev).detach()
    policy_a = (policy_out_dev > 0).long()
    y_dev_selected = torch.gather(y_dev_cf, 1, policy_a)
    dev_policy_val = float(y_dev_selected.mean())
    return dev_policy_val



def train_policy_gmm_benchmark(x, a, y, batch_size, num_stages,
                               max_num_epoch_per_stage, psi_function,
                               policy_network_class,
                               weights_function, norm_matrix_function,
                               nuisance_generator, max_no_improve=None,
                               verbose=False, x_dev=None, a_dev=None,
                               y_dev=None, y_dev_cf=None):
    num_data = x.shape[0]
    dim_x = x.shape[1]
    num_batches = num_data // batch_size + int(num_data % batch_size > 0)

    policy_network = policy_network_class(dim_x=dim_x).double()
    if torch.cuda.is_available():
        policy_network.cuda()
    mean_y_network = nuisance_generator.get_mean_y_network()
    propensity_network = nuisance_generator.get_propensity_network()
    optimizer = torch.optim.Adam(policy_network.parameters(),
                                 weight_decay=0.00)

    for stage in range(num_stages):
        if verbose:
            print("starting stage", stage + 1)
        x_weights_train = weights_function(x)
        policy_out_train = policy_network(x).detach()
        psi_train = psi_function(x, a, y, propensity_network,
                                 mean_y_network).detach()
        phi_train = compute_phi(x_weights_train, policy_out_train, psi_train)
        if stage > 0:
            norm_matrix = norm_matrix_function(phi_train)
        else:
            norm_matrix = torch.eye(phi_train.shape[1]).double()
            if torch.cuda.is_available():
                norm_matrix = norm_matrix.cuda()

        # update first with LBFGS
        lbfgs = torch.optim.LBFGS(policy_network.parameters())

        def closure():
            lbfgs.zero_grad()
            psi_ = psi_function(x, a, y, propensity_network, mean_y_network)
            policy_out_ = policy_network(x)
            x_weights_ = weights_function(x)
            loss_ = gmm_benchmark_loss(x_weights_, policy_out_, psi_,
                                       norm_matrix)
            loss_.backward()
            return loss_
        lbfgs.step(closure)

        min_dev_loss = float("inf")
        no_improve = 0

        for epoch in range(max_num_epoch_per_stage):

            if max_no_improve:
                policy_out_dev = policy_network(x_dev).detach()
                psi_dev = psi_function(x_dev, a_dev, y_dev, propensity_network,
                                       mean_y_network)
                x_weights_dev = weights_function(x_dev)
                dev_loss = gmm_benchmark_loss(x_weights_dev, policy_out_dev,
                                              psi_dev, norm_matrix)

                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        # print("broke after %d epochs" % epoch)
                        break

            if verbose and epoch % 50 == 0:
                dev_policy_val = evaluate_policy_dev(
                    policy_network, x_dev, y_dev_cf)
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
                x_weights = weights_function(x_batch)
                loss = gmm_benchmark_loss(x_weights, policy_out, psi,
                                          norm_matrix)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if verbose:
            dev_policy_val = evaluate_policy_dev(
                policy_network, x_dev, y_dev_cf)
            print("dev policy val: %.4f" % dev_policy_val)

    return policy_network


