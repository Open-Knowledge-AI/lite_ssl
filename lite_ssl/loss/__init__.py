import torch

import lite_ssl.loss.custom as custom


def get_loss(loss_name):
    if hasattr(torch.nn, loss_name):
        return getattr(torch.nn, loss_name)
    elif hasattr(custom, loss_name):
        return getattr(custom, loss_name)
    else:
        raise ValueError(f"Loss {loss_name} not found in torch.nn and ml.loss.custom modules.")
