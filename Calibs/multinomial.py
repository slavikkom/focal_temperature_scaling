from __future__ import division

"""
Multinomial regression core for Dirichlet calibration.

Adapted from https://github.com/dirichletcal/dirichlet_python .
The main local adaptation is replacing the original JAX autodiff/optimization
path with a PyTorch `torch.optim.LBFGS` implementation.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin


class MultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, weights_0=None, method='Full', initializer='identity',
                 reg_format=None, reg_lambda=0.0, reg_mu=None, reg_norm=False,
                 ref_row=True, optimizer='auto', max_iter=int(1024),
                 lr=1.0, device='auto'):
        """
        Params:
            optimizer: string ('auto', 'newton', 'fmin_l_bfgs_b', 'lbfgs')
                Kept for compatibility with the original implementation. All
                supported values now use torch.optim.LBFGS.
        """
        if method not in ['Full', 'Diag', 'FixDiag']:
            raise(ValueError(f"method {method} not avaliable"))

        self.weights_0 = weights_0
        self.method = method
        self.initializer = initializer
        self.reg_format = reg_format
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu  # If number, then ODIR is applied
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.lr = lr
        self.device = device

    def __setup(self):
        self.classes = None
        self.weights_ = self.weights_0
        self.weights_0_ = self.weights_0

    @property
    def coef_(self):
        return self.weights_[:, :-1]

    @property
    def intercept_(self):
        return self.weights_[:, -1]

    def predict_proba(self, S):

        S = _as_numpy(S)
        S_ = np.hstack((S, np.ones((len(S), 1))))

        return np.asarray(_calculate_outputs(self.weights_, S_))

    # FIXME Should we change predict for the argmax?
    def predict(self, S):

        return np.asarray(self.predict_proba(S))

    def fit(self, X, y, *args, **kwargs):

        self.__setup()

        device = _resolve_device(self.device)
        X_t = _as_torch(X, device)
        X_ = torch.hstack([
            X_t,
            torch.ones((len(X_t), 1), dtype=X_t.dtype, device=X_t.device)
        ])
        y_np = _as_numpy(y).ravel()

        self.classes = np.unique(y_np)

        k = len(self.classes)

        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (k * (k + 1))
            else:
                self.reg_lambda = self.reg_lambda / (k * (k - 1))
                self.reg_mu = self.reg_mu / k

        target = _one_hot(y_np, self.classes)

        logging.debug(self.method)

        self.weights_0_ = self._get_initial_weights(self.initializer)

        if self.optimizer in ['auto', 'newton', 'fmin_l_bfgs_b', 'lbfgs',
                              'torch_lbfgs']:
            weights = _torch_lbfgs_update(
                self.weights_0_, X_, target, k, self.method,
                maxiter=self.max_iter, reg_lambda=self.reg_lambda,
                reg_mu=self.reg_mu, ref_row=self.ref_row,
                initializer=self.initializer, reg_format=self.reg_format,
                lr=self.lr, device=device)
        else:
            raise(ValueError('Unknown optimizer: {}'.format(self.optimizer)))

        self.weights_ = _get_weights(weights, k, self.ref_row, self.method)

        return self

    def _get_initial_weights(self, ref_row, initializer='identity'):
        ''' Returns an array containing only the weights of the full weight
        matrix.

        '''

        if initializer not in ['identity', None]:
            raise ValueError

        k = len(self.classes)

        weights_0 = self.weights_0_

        if self.weights_0_ is None:
            if initializer == 'identity':
                weights_0 = _get_identity_weights(k, ref_row, self.method)
            else:
                if self.method == 'Full':
                    weights_0 = np.zeros(k * (k + 1))
                elif self.method == 'Diag':
                    weights_0 = np.zeros(2*k)
                elif self.method == 'FixDiag':
                    weights_0 = np.zeros(1)
        else:
            weights_0 = self.weights_0_

        return np.asarray(weights_0, dtype=np.float64)


def _as_numpy(X):
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def _as_torch(X, device, dtype=torch.float64):
    if torch.is_tensor(X):
        return X.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(X, dtype=dtype, device=device)


def _one_hot(y, classes):
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    indices = np.asarray([class_to_index[label] for label in y])
    target = np.zeros((len(y), len(classes)), dtype=np.float64)
    target[np.arange(len(y)), indices] = 1.0
    return target


def _resolve_device(device):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _objective(params, *args):
    (X, _, y, k, method, reg_lambda, reg_mu, ref_row, _, reg_format) = args
    device = torch.device('cpu')
    params_t = torch.as_tensor(params, dtype=torch.float64, device=device)
    X_t = _as_torch(X, device)
    y_t = _as_torch(y, device)
    reg_context = _get_regularization_context(
        X_t, k, reg_mu=reg_mu, reg_format=reg_format)
    loss = _objective_torch(params_t, X_t, y_t, k, method, reg_lambda, reg_mu,
                            ref_row, reg_format, reg_context)
    return float(loss.detach().cpu())


def _gradient(params, *args):
    (X, _, y, k, method, reg_lambda, reg_mu, ref_row, _, reg_format) = args
    device = torch.device('cpu')
    params_t = torch.tensor(params, dtype=torch.float64, device=device,
                            requires_grad=True)
    X_t = _as_torch(X, device)
    y_t = _as_torch(y, device)
    reg_context = _get_regularization_context(
        X_t, k, reg_mu=reg_mu, reg_format=reg_format)
    loss = _objective_torch(params_t, X_t, y_t, k, method, reg_lambda, reg_mu,
                            ref_row, reg_format, reg_context)
    loss.backward()
    return params_t.grad.detach().cpu().numpy()


def _hessian(params, *args):
    (X, _, y, k, method, reg_lambda, reg_mu, ref_row, _, reg_format) = args
    device = torch.device('cpu')
    X_t = _as_torch(X, device)
    y_t = _as_torch(y, device)

    def closure(params_t):
        reg_context = _get_regularization_context(
            X_t, k, reg_mu=reg_mu, reg_format=reg_format)
        return _objective_torch(params_t, X_t, y_t, k, method, reg_lambda,
                                reg_mu, ref_row, reg_format, reg_context)

    params_t = torch.as_tensor(params, dtype=torch.float64, device=device)
    return torch.autograd.functional.hessian(closure, params_t).detach().cpu().numpy()


def _get_regularization_context(X, k, reg_mu, reg_format):
    eye = torch.eye(k, dtype=X.dtype, device=X.device)
    zeros_col = torch.zeros((k, 1), dtype=X.dtype, device=X.device)
    if reg_mu is None:
        if reg_format == 'identity':
            return torch.hstack([eye, zeros_col])
        return torch.zeros((k, k + 1), dtype=X.dtype, device=X.device)

    return torch.hstack([eye, zeros_col])


def _objective_torch(params, X, y, k, method, reg_lambda, reg_mu, ref_row,
                     reg_format, reg_context=None):
    weights = _get_weights_torch(params, k, ref_row, method)
    logits = torch.matmul(X, weights.transpose(0, 1))
    loss = torch.mean(-torch.sum(y * F.log_softmax(logits, dim=1), dim=1))

    if reg_context is None:
        reg_context = _get_regularization_context(
            X, k, reg_mu=reg_mu, reg_format=reg_format)

    if reg_mu is None:
        loss = loss + reg_lambda * torch.sum((weights - reg_context) ** 2)
    else:
        weights_hat = weights * (1.0 - reg_context)
        loss = loss + reg_lambda * torch.sum(weights_hat[:, :-1] ** 2) + \
            reg_mu * torch.sum(weights_hat[:, -1] ** 2)

    return loss


def _get_weights(params, k, ref_row, method):
    '''Reshapes the given params into the full weight matrix.'''

    params = np.asarray(params, dtype=np.float64)

    if method in ['Full', None]:
        raw_weights = params.reshape(-1, k+1)

    elif method == 'Diag':
        raw_weights = np.hstack([np.diag(params[:k]),
                                 params[k:].reshape(-1, 1)])

    elif method == 'FixDiag':
        raw_weights = np.hstack([np.eye(k) * params[0], np.zeros((k, 1))])
    else:
        raise(ValueError(f"Unknown calibration method {method}"))

    if ref_row:
        weights = raw_weights - np.repeat(
            raw_weights[-1, :].reshape(1, -1), k, axis=0)
    else:
        weights = raw_weights

    return weights


def _get_weights_torch(params, k, ref_row, method):
    '''Torch version of _get_weights used during optimization.'''

    if method in ['Full', None]:
        raw_weights = params.reshape(-1, k+1)

    elif method == 'Diag':
        raw_weights = torch.hstack([
            torch.diag(params[:k]),
            params[k:].reshape(-1, 1)
        ])

    elif method == 'FixDiag':
        raw_weights = torch.hstack([
            torch.eye(k, dtype=params.dtype, device=params.device) * params[0],
            torch.zeros((k, 1), dtype=params.dtype, device=params.device)
        ])
    else:
        raise(ValueError(f"Unknown calibration method {method}"))

    if ref_row:
        weights = raw_weights - raw_weights[-1, :].reshape(1, -1).repeat(k, 1)
    else:
        weights = raw_weights

    return weights


def _get_identity_weights(n_classes, ref_row, method):

    raw_weights = None

    if (method is None) or (method == 'Full'):
        raw_weights = np.zeros((n_classes, n_classes + 1)) + \
                      np.hstack([np.eye(n_classes), np.zeros((n_classes, 1))])
        raw_weights = raw_weights.ravel()

    elif method == 'Diag':
        raw_weights = np.hstack([np.ones(n_classes), np.zeros(n_classes)])

    elif method == 'FixDiag':
        raw_weights = np.ones(1)

    return raw_weights.ravel()


def _calculate_outputs(weights, X):
    mul = np.dot(X, weights.transpose())
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - np.max(X, axis=1).reshape(-1, 1)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)


def _torch_lbfgs_update(weights_0, X, target, k, method_, maxiter=int(1024),
                        ftol=1e-12, gtol=1e-8, reg_lambda=0.0, reg_mu=None,
                        ref_row=True, initializer=None, reg_format=None,
                        lr=1.0, device='auto'):

    del initializer

    device = _resolve_device(device)
    X_t = _as_torch(X, device)
    target_t = _as_torch(target, device)
    weights = torch.tensor(weights_0, dtype=torch.float64, device=device,
                           requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [weights], lr=lr, max_iter=maxiter, tolerance_grad=gtol,
        tolerance_change=ftol, line_search_fn='strong_wolfe')

    reg_context = _get_regularization_context(
        X_t, k, reg_mu=reg_mu, reg_format=reg_format)

    def closure():
        optimizer.zero_grad()
        loss = _objective_torch(weights, X_t, target_t, k, method_,
                                reg_lambda, reg_mu, ref_row, reg_format,
                                reg_context)
        loss.backward()
        return loss

    optimizer.step(closure)

    final_loss = _objective_torch(weights, X_t, target_t, k, method_,
                                  reg_lambda, reg_mu, ref_row, reg_format,
                                  reg_context)
    logging.debug("%s: final log-loss = %.7e", method_, float(final_loss))

    return weights.detach().cpu().numpy()


def _newton_update(weights_0, X, XX_T, target, k, method_, maxiter=int(1024),
                   ftol=1e-12, gtol=1e-8, reg_lambda=0.0, reg_mu=None,
                   ref_row=True, initializer=None, reg_format=None):

    del XX_T
    return _torch_lbfgs_update(
        weights_0, X, target, k, method_, maxiter=maxiter, ftol=ftol,
        gtol=gtol, reg_lambda=reg_lambda, reg_mu=reg_mu, ref_row=ref_row,
        initializer=initializer, reg_format=reg_format)
