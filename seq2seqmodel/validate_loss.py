import torch
import torch.nn as nn
import math

log_input = True
full = True

def validate_loss(output, target, flag, full, eps=1e-08):
    val = 0
    for li_x, li_y in zip(output, target):
        for i, xy in enumerate(zip(li_x, li_y)):
            x, y = xy
            if flag:
                loss_val = math.exp(x) - y * x
                if full:
                    if y <= 1:
                        loss_val = math.exp(x) - y * x + 0
                    else:
                        loss_val = math.exp(x) - y * x + \
                                   y * math.log(y) - y + 0.5 * math.log(2 * math.pi * y)
            else:
                loss_val = x - y * math.log(x + eps)
                if full:
                    if y <= 1:
                        loss_val = x - y * math.log(x + eps) + 0
                    else:
                        loss_val = x - y * math.log(x + eps) + \
                                   y * math.log(y) - y + 0.5 * math.log(2 * math.pi * y)
            val += loss_val
    return val / output.nelement()



