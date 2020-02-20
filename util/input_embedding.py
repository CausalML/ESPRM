import torch


def to_quadratic(x):
    batch_size = x.shape[0]
    x_squared = (x.unsqueeze(1) * x.unsqueeze(2)).view(batch_size, -1)
    sym = torch.DoubleTensor([[1, 0, 0], [0, 1, 0],
                              [0, 1, 0], [0, 0, 1]])
    if torch.cuda.is_available():
        sym = sym.cuda()
    return torch.cat([x, torch.matmul(x_squared, sym)], dim=1)
