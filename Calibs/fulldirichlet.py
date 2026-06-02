"""
Full Dirichlet calibration.

Adapted from https://github.com/dirichletcal/dirichlet_python .
The main local adaptation is using PyTorch instead of JAX for the optimization
backend in `multinomial.py`.
"""

from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
from sklearn.metrics import log_loss

try:
    from .multinomial import MultinomialRegression
    from .utils import as_numpy, copy_array, log_clipped
except ImportError:
    from multinomial import MultinomialRegression
    from utils import as_numpy, copy_array, log_clipped


class FullDirichletCalibrator(ClassifierMixin, BaseEstimator):
    def __init__(self, reg_lambda=0.0, reg_mu=None, weights_init=None,
                 initializer='identity', reg_norm=False, ref_row=True,
                 optimizer='auto', max_iter=int(1024), lr=1.0,
                 device='auto'):

        """
        Params:
            weights_init: (nd.array) weights used for initialisation, if None
            then idendity matrix used. Shape = (n_classes - 1, n_classes + 1)
            comp_l2: (bool) If true, then complementary L2 regularization used
            (off-diagonal regularization)
            optimizer: string ('auto', 'newton', 'fmin_l_bfgs_b', 'lbfgs')
                Kept for compatibility with the original implementation. All
                supported values use torch.optim.LBFGS in this port.
        """
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu  # Complementary L2 regularization. (Off-diagonal)
        self.weights_init = weights_init  # Input weights for initialisation
        self.initializer = initializer
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.lr = lr
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        self.weights_ = self.weights_init

        if X_val is None:
            X_val = copy_array(X)
            y_val = copy_array(y)
        else:
            X_val = copy_array(X_val)
            y_val = copy_array(y_val)

        _X = log_clipped(copy_array(X))
        _X_val = log_clipped(copy_array(X_val))

        self.calibrator_ = MultinomialRegression(
            method='Full', reg_lambda=self.reg_lambda, reg_mu=self.reg_mu,
            reg_norm=self.reg_norm, ref_row=self.ref_row,
            optimizer=self.optimizer, max_iter=self.max_iter, lr=self.lr,
            device=self.device)
        self.calibrator_.fit(_X, y, *args, **kwargs)
        self.classes_ = self.calibrator_.classes
        self.final_loss = log_loss(
            as_numpy(y_val), self.calibrator_.predict_proba(_X_val))

        return self

    @property
    def weights(self):
        if self.calibrator_ is not None:
            return self.calibrator_.weights_
        return self.weights_init

    @property
    def cannonical_weights(self):
        b = self.weights[:, -1]
        w = self.weights[:, :-1]
        col_min = np.min(w, axis=0)
        a = w - col_min

        def softmax(z):
            return np.divide(np.exp(z), np.sum(np.exp(z)))

        c = softmax(np.matmul(w, np.log(np.ones(len(b)) / len(b))) + b)
        return np.hstack((a, c.reshape(-1, 1)))

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = log_clipped(S)
        return np.asarray(self.calibrator_.predict_proba(S))

    def predict(self, S):
        S = log_clipped(S)
        return np.asarray(self.calibrator_.predict(S))
