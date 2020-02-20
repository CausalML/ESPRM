

def tensor_invalid(x):
    x = x.view(-1)
    tensor_len = len(x)
    num_greater_ninf = int((x > float("-inf")).sum())
    num_inf = int((x == float("inf")).sum())
    num_valid = num_greater_ninf - num_inf
    # print(num_valid, num_inf, num_greater_ninf, tensor_len, x[:100])
    return tensor_len != num_valid