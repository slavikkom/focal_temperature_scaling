from fulldirichlet import FullDirichletCalibrator
from diagdirichlet import DiagonalDirichletCalibrator
from betadirichlet import BetaDirichletCalibrator
from multinomial import MultinomialRegression
from utils import log_clipped

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     GridSearchCV,
                                     cross_val_score)

from sklearn.metrics import log_loss
def neg_log_loss_scorer(estimator, X, y):
    return -log_loss(y, estimator.predict_proba(X))

dataset = load_iris()
x = dataset['data']
y = dataset['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,
                                                    test_size=0.3)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

classifier = GaussianNB()
print('Training a classifier with cross-validation')
scores = cross_val_score(classifier, x_train, y_train, cv=skf,
                         scoring='neg_log_loss')
print('Crossval scores: {}'.format(scores))
print('Average neg log loss {:.3f}'.format(scores.mean()))
classifier.fit(x_train, y_train)

cla_scores_train = classifier.predict_proba(x_train)
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

cla_scores_test = classifier.predict_proba(x_test)

calibration_method = 'diagonal'
# Options:
# calibration_method = 'full'
# calibration_method = 'odir'
# calibration_method = 'diagonal'
# calibration_method = 'fixdiag'
# calibration_method = 'beta_dirichlet'

if calibration_method == 'beta_dirichlet':
    # BetaDirichletCalibrator is a lightweight custom calibrator, not a
    # sklearn estimator, so this example fits it directly instead of using
    # GridSearchCV.
    calibrator = BetaDirichletCalibrator(lambda_l2=1e-3, maxiter=100)
    calibrator.fit(cla_scores_train, y_train)
    cal_scores_test = calibrator.predict_proba(cla_scores_test)
    print('Fitted BetaDirichletCalibrator with lambda_l2={}'.format(
        calibrator.lambda_l2))
elif calibration_method == 'fixdiag':
    # FixDiag operates directly on log-probability features. It learns one
    # shared scale alpha: softmax(alpha * log(p)).
    calibrator = MultinomialRegression(
        method='FixDiag', reg_lambda=1e-3, max_iter=100)
    calibrator.fit(log_clipped(cla_scores_train), y_train)
    cal_scores_test = calibrator.predict_proba(log_clipped(cla_scores_test))
    print('Fitted FixDiag calibrator with reg_lambda={}'.format(
        calibrator.reg_lambda))
else:
    if calibration_method == 'full':
        calibrator = FullDirichletCalibrator()
        param_grid = {'reg_lambda': reg, 'reg_mu': [None]}
    elif calibration_method == 'odir':
        calibrator = FullDirichletCalibrator()
        param_grid = {'reg_lambda': reg, 'reg_mu': reg}
    elif calibration_method == 'diagonal':
        calibrator = DiagonalDirichletCalibrator()
        param_grid = {'reg_lambda': reg, 'reg_mu': [None]}
    else:
        raise ValueError('Unknown calibration_method: {}'.format(
            calibration_method))

    gscv = GridSearchCV(
        calibrator, param_grid=param_grid, cv=skf, scoring='neg_log_loss')

    # gscv = GridSearchCV(calibrator, param_grid=param_grid,
    #                     cv=skf, scoring=neg_log_loss_scorer)

    gscv.fit(cla_scores_train, y_train)

    print('Grid of parameters cross-validated')
    print(gscv.param_grid)
    print('Best parameters: {}'.format(gscv.best_params_))
    cal_scores_test = gscv.predict_proba(cla_scores_test)

cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.2f}, calibrator {:.2f}".format(
    cla_loss, cal_loss))

print("before calibration:\n {}".format(cla_scores_test[:5].round(2)))
print("after calibration:\n {}".format(cal_scores_test[:5].round(2)))
