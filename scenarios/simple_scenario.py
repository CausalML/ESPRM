import numpy as np
import torch

from policy_learning.policy_networks import LinearPolicyNetwork


class GenericSimpleScenario(object):
    def __init__(self, **setup_args):
        object.__init__(self)
        self.set_params(**setup_args)

    def set_params(self, x_mean, x_std, x_dim, y_theta_0, y_theta_1,
                   y_std, a_theta, y_b_0=0, y_b_1=0, a_b=0):
        self.x_mean = x_mean
        self.x_std = x_std
        self.x_dim = x_dim

        self.y_theta_0 = torch.from_numpy(y_theta_0).double()
        self.y_theta_1 = torch.from_numpy(y_theta_1).double()
        self.a_theta = torch.from_numpy(a_theta).double()
        if torch.cuda.is_available():
            self.y_theta_0 = self.y_theta_0.cuda()
            self.y_theta_1 = self.y_theta_1.cuda()
            self.a_theta = self.a_theta.cuda()
        self.y_std = y_std.reshape(1, -1)

        self.y_b_0 = y_b_0
        self.y_b_1 = y_b_1
        self.a_b = a_b

    def initialize(self):
        pass

    def _mean_y(self, x):
        y_mean_0 = torch.matmul(x, self.y_theta_0) + self.y_b_0
        y_mean_1 = torch.matmul(x, self.y_theta_1) + self.y_b_1
        return torch.stack([y_mean_0, y_mean_1], dim=1)

    def get_y_network(self):
        return self._mean_y

    def _propensity(self, x):
        a_probs = torch.sigmoid(torch.matmul(x, self.a_theta) + self.a_b)
        return torch.stack([a_probs, 1 - a_probs], dim=1)

    def get_propensity_network(self):
        return self._propensity

    def get_optimal_policy(self, policy_network_class, x_val, y_cf_val):
        policy = LinearPolicyNetwork(self.x_dim).double()
        if torch.cuda.is_available():
            policy = policy.cuda()
        theta_diff = (self.y_theta_1 - self.y_theta_0).reshape(1, -1)
        bias_diff = torch.DoubleTensor([self.y_b_1 - self.y_b_0])
        if torch.cuda.is_available():
            bias_diff = bias_diff.cuda()
        policy.linear.weight.data = theta_diff
        policy.linear.bias.data = bias_diff
        return policy

    def get_theta_dict(self):
        return {
            "y_theta_0": self._get_theta(self.y_theta_0, self.y_b_0),
            "y_theta_1": self._get_theta(self.y_theta_1, self.y_b_1),
            "a_theta": self._get_theta(self.a_theta, self.a_b),
        }

    @staticmethod
    def _get_theta(weights, bias):
        weights_np = weights.cpu().numpy()
        p_list = [str(bias)] + [str(w) for w in weights_np]
        return ",".join(p_list)

    def sample_data(self, num_data):
        x_dim = self.x_mean.shape[0]
        x = np.random.normal(self.x_mean, self.x_std.reshape(1, -1),
                             size=(num_data, x_dim))
        x = torch.from_numpy(x).double()
        if torch.cuda.is_available():
            x = x.cuda()

        y_mean = self._mean_y(x).cpu().detach().numpy()
        y_0 = np.random.normal(y_mean[:, 0], self.y_std).reshape(-1, 1)
        y_1 = np.random.normal(y_mean[:, 1], self.y_std).reshape(-1, 1)
        y_cf = np.concatenate([y_0, y_1], axis=1)

        a_probs = self._propensity(x).cpu().detach().numpy()
        a = np.array([np.random.choice([0, 1], p=a_probs[i])
                      for i in range(num_data)])
        y = np.array([y_cf[i, a[i]] for i in range(num_data)])

        a = torch.from_numpy(a).long()
        y = torch.from_numpy(y).double().view(-1, 1)
        y_cf = torch.from_numpy(y_cf).double()
        if torch.cuda.is_available():
            a = a.cuda()
            y = y.cuda()
            y_cf = y_cf.cuda()

        return x, a, y, y_cf


class SimpleScenario(GenericSimpleScenario):
    def __init__(self):
        setup_args = {
            "x_mean": np.zeros(2),
            "x_std": np.ones(2),
            "x_dim": 2,
            "y_theta_0": np.array([1.0, -1.0]),
            "y_theta_1": np.array([-2.0, 0.5]),
            "y_std": np.ones(1) * 1,
            "a_theta": np.array([1.0, 2.0]),
            "y_b_0": 0,
            "y_b_1": 0,
            "a_b": 0,
        }

        GenericSimpleScenario.__init__(self, **setup_args)


class RandomSimpleScenario(GenericSimpleScenario):
    def __init__(self):
        setup_args = RandomSimpleScenario._get_random_setup_args()
        GenericSimpleScenario.__init__(self, **setup_args)

    def initialize(self):
        setup_args = RandomSimpleScenario._get_random_setup_args()
        self.set_params(**setup_args)

    @staticmethod
    def _get_random_setup_args():
        setup_args = {
            "x_mean": np.zeros(2),
            "x_std": np.ones(2),
            "x_dim": 2,
            "y_theta_0": np.random.randn(2),
            "y_theta_1": np.random.randn(2),
            "y_std": np.ones(1) * 1,
            "a_theta": np.random.randn(2),
            "y_b_0": float(np.random.randn(1)),
            "y_b_1": float(np.random.randn(1)),
            "a_b": float(np.random.randn(1)),
        }
        return setup_args


