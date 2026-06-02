import numpy as np
import torch


def as_numpy(X):
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def copy_array(X):
    if torch.is_tensor(X):
        return X.clone()
    return X.copy()


def clip_for_log(X):
    if torch.is_tensor(X):
        if not torch.is_floating_point(X):
            X = X.float()
        eps = torch.finfo(X.dtype).tiny
        return torch.clamp(X, eps, 1 - eps)
    X = np.asarray(X)
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1 - eps)


def clip(X):
    return clip_for_log(X)


def log_clipped(X):
    X = clip_for_log(X)
    if torch.is_tensor(X):
        return torch.log(X)
    return np.log(X)


def clip_jax(X):
    return clip(X)
