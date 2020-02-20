import torch
import torch.nn.functional as F
import numpy as np
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import FloatVector


class LinearFunction(object):
    def __init__(self, intercept, weights):
        self.intercept = intercept
        self.weights = weights
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def __call__(self, x):
        out = torch.matmul(x, self.weights) + self.intercept
        return out.view(-1, 1)


class LogisticFunction(LinearFunction):
    def __init__(self, intercept, weights):
        LinearFunction.__init__(self, intercept, weights)

    def __call__(self, x):
        out = torch.sigmoid(torch.matmul(x, self.weights) + self.intercept)
        return out.view(-1, 1)


def r_logistic_regression(x, y):
    """
    Assume x and y are pytorch tensors
    """
    x_dim = x.shape[1]
    for i in range(x_dim):
        xr_i = FloatVector(x[:, i])
        robjects.globalenv["x%d" % i] = xr_i
    yr = FloatVector(y.view(-1))
    robjects.globalenv["y"] = yr
    formula = "y~" + "+".join("x%d" % i_ for i_ in range(x_dim))
    robjects.r("fit <- glm(%s, family=binomial())" % formula)
    robjects.r("lin_reg_out <- coef(fit)")
    reg_out = robjects.r["lin_reg_out"]
    intercept = float(reg_out[0])
    weights = torch.DoubleTensor(reg_out[1:])
    weights[torch.isnan(weights)] = 0.0
    return LogisticFunction(intercept, weights)


def r_linear_regression(x, y):
    """
    Assume x and y are pytorch tensors
    """
    x_dim = x.shape[1]
    for i in range(x_dim):
        xr_i = FloatVector(x[:, i])
        robjects.globalenv["x%d" % i] = xr_i
    yr = FloatVector(y.view(-1))
    robjects.globalenv["y"] = yr
    formula = "y~" + "+".join("x%d" % i_ for i_ in range(x_dim))
    robjects.r("fit <- lm(%s)" % formula)
    robjects.r("lin_reg_out <- coef(fit)")
    lin_reg_out = robjects.r["lin_reg_out"]
    intercept = float(lin_reg_out[0])
    weights = torch.DoubleTensor(lin_reg_out[1:])
    weights[torch.isnan(weights)] = 0.0
    return LinearFunction(intercept, weights)


def debug():
    x = torch.randn(100000, 3).double()
    w = torch.DoubleTensor([1.0, -1.0, 2.0])
    b = 1.5

    # y_prob = torch.sigmoid(torch.matmul(x, w) + b).view(-1)
    # y = [np.random.choice([1, 0], p=(p, 1-p)) for p in y_prob]
    # y = torch.LongTensor(y).view(-1, 1)
    # logistic_reg = r_logistic_regression(x, y)
    # print(logistic_reg.intercept)
    # print(logistic_reg.weights)
    # print(logistic_reg(x[:10]))
    # print(y[:10])

    y = (torch.matmul(x, w) + b).view(-1, 1)
    linear_reg = r_linear_regression(x, y)
    print(linear_reg.intercept)
    print(linear_reg.weights)
    print(linear_reg.intercept)
    print(linear_reg.weights)
    print(linear_reg(x[:10]))
    print(y[:10])



if __name__ == "__main__":
    debug()
