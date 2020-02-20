import torch
import torch.nn as nn
import torch.nn.functional as F
from util.input_embedding import to_quadratic


class LinearPolicyNetwork(nn.Module):
    def __init__(self, dim_x):
        nn.Module.__init__(self)
        self.dim_x = dim_x
        self.linear = nn.Linear(self.dim_x, 1)

    def forward(self, x):
        return self.linear(x)

    def get_policy_weights(self):
        weights = self.linear.weight.data.cpu().numpy()
        bias = self.linear.bias.data.cpu().numpy()
        p_list = [bias[0]] + [w for w in weights[0]]
        return ",".join([str(w) for w in p_list])

    def get_negative(self):
        negative_policy = LinearPolicyNetwork(self.dim_x).double()
        if torch.cuda.is_available():
            negative_policy = negative_policy.cuda()
        negative_policy.linear.weight.data = -self.linear.weight.data
        negative_policy.linear.bias.data = -self.linear.bias.data
        return negative_policy


class QuadraticPolicyNetwork(nn.Module):
    def __init__(self, dim_x):
        nn.Module.__init__(self)
        self.dim_x = dim_x
        self.input_size = 2 * dim_x + dim_x * (dim_x - 1) // 2
        self.linear = nn.Linear(self.input_size, 1)

    def forward(self, x):
        return self.linear(to_quadratic(x))

    def get_policy_weights(self):
        weights = self.linear.weight.data.cpu().numpy()
        bias = self.linear.bias.data.cpu().numpy()
        p_list = [bias[0]] + [w for w in weights[0]]
        return ",".join([str(w) for w in p_list])

    def get_negative(self):
        negative_policy = QuadraticPolicyNetwork(self.dim_x).double()
        if torch.cuda.is_available():
            negative_policy = negative_policy.cuda()
        negative_policy.linear.weight.data = -self.linear.weight.data
        negative_policy.linear.bias.data = -self.linear.bias.data
        return negative_policy


class FlexiblePolicyNetwork(nn.Module):
    def __init__(self, dim_x):
        nn.Module.__init__(self)
        self.dim_x = dim_x
        self.linear_1 = nn.Linear(self.dim_x, 50)
        self.linear_2 = nn.Linear(50, 1)

    def forward(self, x):
        h = self.linear_1(x)
        # return self.linear_2(torch.tanh(h))
        return self.linear_2(F.leaky_relu(h))

    def get_policy_weights(self):
        return None

    def get_negative(self):
        negative_policy = FlexiblePolicyNetwork(self.dim_x).double()
        if torch.cuda.is_available():
            negative_policy = negative_policy.cuda()
        negative_policy.linear_2.weight.data = -self.linear_2.weight.data
        negative_policy.linear_2.bias.data = -self.linear_2.bias.data
        return negative_policy


def debug():
    n = LinearPolicyNetwork(2)
    print(n.get_policy_weights())


if __name__ == "__main__":
    debug()
