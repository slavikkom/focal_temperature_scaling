from sklearn.metrics import log_loss

try:
    from .multinomial import MultinomialRegression
    from .fulldirichlet import FullDirichletCalibrator
    from .utils import as_numpy, copy_array, log_clipped
except ImportError:
    from multinomial import MultinomialRegression
    from fulldirichlet import FullDirichletCalibrator
    from utils import as_numpy, copy_array, log_clipped


class DiagonalDirichletCalibrator(FullDirichletCalibrator):
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
            method='Diag', reg_lambda=self.reg_lambda, reg_mu=self.reg_mu,
            reg_norm=self.reg_norm, ref_row=self.ref_row,
            optimizer=self.optimizer, max_iter=self.max_iter, lr=self.lr,
            device=self.device)
        self.calibrator_.fit(_X, y, *args, **kwargs)
        self.classes_ = self.calibrator_.classes
        self.final_loss = log_loss(
            as_numpy(y_val), self.calibrator_.predict_proba(_X_val))

        return self
