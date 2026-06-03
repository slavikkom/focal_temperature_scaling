import torch
from torch import nn
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from temperature_scaling import ModelWithTemperature
from Metrics.metrics import AdaptiveECELoss, SmoothECELoss
from temperature_scaling import multi_focal_link
from new_links import linear_invlink, multi_link
from Calibs.fulldirichlet import FullDirichletCalibrator

link_dict = {
    'softmax': [1],
    'focal': [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7],
    'focal_linear': [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7],
    'exp_p': [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7],
    'exp_1mp': [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7],
    'one_minus_power': [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7],
    'generalized_focal': [(b, g) for b in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7] for g in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7]],
    'log_power': [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7]
}

def multi_acc(y_pred, y_test):
    y_pred = torch.argmax(y_pred, dim = 1)    
    
    correct_pred = (y_pred == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = acc.cpu().numpy() * 100
    
    return acc


def evaluate(y_true, probs, num_classes=10):
    
    epoch_loss = {}
        
    mse_loss = torch.nn.MSELoss()

    log_loss_criterion = nn.NLLLoss()
        
    calibration_criterion = AdaptiveECELoss(return_stats=True)
    smooth_calibration_criterion = SmoothECELoss(return_sigma=True)
    fixed_smooth_calibration_criterion = SmoothECELoss(bandwidth=0.05)
        
    epoch_loss['CE'] = log_loss_criterion(torch.log(probs), y_true.long()).item()
    epoch_loss['ECE'] = calibration_criterion(probs, y_true.long())
    sm_ece, sm_ece_sigma = smooth_calibration_criterion(probs, y_true.long())
    epoch_loss['smECE'] = sm_ece.item()
    epoch_loss['smECE_sigma'] = sm_ece_sigma
    epoch_loss['smECE_0.05'] = fixed_smooth_calibration_criterion(probs, y_true.long()).item()
        
    epoch_loss['Brier'] = mse_loss(probs, torch.nn.functional.one_hot(y_true.long(), num_classes=num_classes).float()).item()
    epoch_loss['ACC'] = multi_acc(probs, y_true)

    return epoch_loss
    
def get_probs(logits, T=1, a=1, link='softmax'):
    if link == 'softmax':
        probs = torch.nn.Softmax(dim=1)(logits / T)
    elif link == 'focal':
        probs = multi_focal_link(logits / T, a)
    elif link == 'generalized_focal':
        probs = multi_link(logits / T, link, a[0], a[1])
    else:
        probs = multi_link(logits / T, link, a)
    return probs


def _probs_to_tensor(probs, device):
    return torch.as_tensor(probs, dtype=torch.float32, device=device)


def _to_numpy_array(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _add_missing_class_examples(probs, labels, num_classes):
    present_classes = set(np.asarray(labels).astype(int).ravel().tolist())
    missing_classes = [
        class_idx for class_idx in range(num_classes)
        if class_idx not in present_classes
    ]
    if not missing_classes:
        return probs, labels

    eps = 1e-6
    pseudo_probs = np.full((len(missing_classes), num_classes), eps)
    pseudo_probs[np.arange(len(missing_classes)), missing_classes] = (
        1.0 - eps * (num_classes - 1)
    )
    probs = np.vstack([probs, pseudo_probs.astype(probs.dtype, copy=False)])
    labels = np.concatenate([labels, np.asarray(missing_classes, dtype=labels.dtype)])
    return probs, labels


def _dirichlet_key(reg_lambda, reg_mu):
    return "lambda_{:.0e}_mu_{:.0e}".format(reg_lambda, reg_mu)


def _compact_cv_results(cv_results):
    compact = {}
    params = cv_results["params"]
    means = cv_results["mean_test_score"]
    stds = cv_results["std_test_score"]
    ranks = cv_results["rank_test_score"]

    for params_i, mean_i, std_i, rank_i in zip(params, means, stds, ranks):
        reg_lambda = params_i["reg_lambda"]
        reg_mu = params_i["reg_mu"]
        compact[_dirichlet_key(reg_lambda, reg_mu)] = {
            "reg_lambda": float(reg_lambda),
            "reg_mu": float(reg_mu),
            "mean_test_neg_log_loss": float(mean_i),
            "std_test_neg_log_loss": float(std_i),
            "rank_test_neg_log_loss": int(rank_i),
        }
    return compact


def dirichlet_calibration_evaluation(val_logits, val_labels, test_logits, test_labels,
                                     num_classes=10, device='cuda',
                                     train_logits=None, train_labels=None,
                                     reg_grid=None, cv_folds=3, seed=1, smoke_test=False):
    if reg_grid is None:
        reg_grid = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    val_probs = get_probs(val_logits, T=1, a=1, link='softmax')
    test_probs = get_probs(test_logits, T=1, a=1, link='softmax')
    val_labels_cv = val_labels.long()
    val_probs_cv = _to_numpy_array(val_probs)
    test_probs_cv = _to_numpy_array(test_probs)
    val_labels_cv_np = _to_numpy_array(val_labels_cv)

    if smoke_test:
        smoke_reg_lambda = 1e-3
        smoke_reg_mu = 0.0
        # a rememdy since dirichlet implementation requires the number of unique classes
        # within the val_labels array to be equal to the number of classes specified by num_classes, 
        # so we add one example for any missing class with a very small probability to avoid affecting the results.
        # this hack is only for the smoke test to run without errors and should not affect the results
        fit_probs_cv, fit_labels_cv_np = _add_missing_class_examples(
            val_probs_cv, val_labels_cv_np, num_classes
        )
        estimator = FullDirichletCalibrator(
            reg_lambda=smoke_reg_lambda,
            reg_mu=smoke_reg_mu,
            max_iter=25,
        ).fit(fit_probs_cv, fit_labels_cv_np)
        best_params = {"reg_lambda": 1e-3, "reg_mu": 0}
        best_score = -float(estimator.final_loss)
        n_splits = 0
        selection_metric = "train_neg_log_loss_smoke_test"
        param_grid = {
            "reg_lambda": [smoke_reg_lambda],
            "reg_mu": [smoke_reg_mu],
        }
        cv_results = {}
    else:
        # how many examples per class in the validation set? We need at least 2 for each class to do cross-validation, 
        # and we can't have more folds than the smallest class count. 
        # So we compute the class counts and determine the number of splits accordingly.
        class_counts = torch.bincount(val_labels_cv, minlength=num_classes)
        nonzero_class_counts = class_counts[class_counts > 0]
        min_class_count = int(nonzero_class_counts.min().item()) if nonzero_class_counts.numel() else 0
        n_splits = min(cv_folds, min_class_count)
        if n_splits < 2:
            raise ValueError(
                "Dirichlet GridSearchCV needs at least two validation examples per "
                "present class; smallest present class count is {}.".format(min_class_count)
            )

        param_grid = {
            "reg_lambda": reg_grid,
            "reg_mu": reg_grid,
        }
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(
            FullDirichletCalibrator(),
            param_grid=param_grid,
            cv=cv,
            scoring="neg_log_loss",
            refit=True,
        )
        grid_search.fit(val_probs_cv, val_labels_cv_np)

        estimator = grid_search
        best_params = grid_search.best_params_
        best_score = float(grid_search.best_score_)
        selection_metric = "neg_log_loss"
        cv_results = _compact_cv_results(grid_search.cv_results_)
    result = {
        "selection_metric": selection_metric,
        "cv_folds": n_splits,
        "param_grid": {
            "reg_lambda": [float(v) for v in param_grid["reg_lambda"]],
            "reg_mu": [float(v) for v in param_grid["reg_mu"]],
        },
        "best": {
            "key": _dirichlet_key(best_params["reg_lambda"], best_params["reg_mu"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "reg_mu": float(best_params["reg_mu"]),
            "mean_test_neg_log_loss": best_score,
        },
        "cv_results": cv_results,
    }

    val_cal_probs = _probs_to_tensor(estimator.predict_proba(val_probs_cv), device)
    test_cal_probs = _probs_to_tensor(estimator.predict_proba(test_probs_cv), device)
    result["val"] = evaluate(val_labels, val_cal_probs, num_classes=num_classes)
    result["test"] = evaluate(test_labels, test_cal_probs, num_classes=num_classes)

    if train_logits is not None and train_labels is not None:
        train_probs = get_probs(train_logits, T=1, a=1, link='softmax')
        train_probs_cv = _to_numpy_array(train_probs)
        train_cal_probs = _probs_to_tensor(estimator.predict_proba(train_probs_cv), device)
        result["train"] = evaluate(train_labels, train_cal_probs, num_classes=num_classes)

    return {
        "softmax": {
            "full_odir": result,
        }
    }

def focal_calibration_evaluation(net, val_logits, val_labels, test_logits, test_labels, num_classes=10, device='cuda', train_logits=None, train_labels=None, links=None):
    if links is None or "all" in links:
        active_link_dict = link_dict
    else:
        unknown_links = [link for link in links if link not in link_dict]
        if unknown_links:
            raise ValueError(
                "Unknown calibration link(s): {}. Valid links are: {}".format(
                    ", ".join(unknown_links), ", ".join(link_dict.keys())
                )
            )
        active_link_dict = {link: link_dict[link] for link in links}

    calib_states = ('calibrated', 'uncalibrated')
    datasets = ['val', 'test']
    if train_logits is not None:
        datasets.insert(0, 'train')   # now ['train','val','test']

    evaluation_metrics = {
        phase: { state: {} for state in calib_states }
        for phase in datasets
    }

    gamma_dict = {}
    
    for data_set in datasets:
        evaluation_metrics[data_set]['uncalibrated'] = {}
        for T_metric in ['ce', 'ece']:
            evaluation_metrics[data_set]['calibrated'][T_metric] = {}
            for link_name in active_link_dict:
                evaluation_metrics[data_set]['calibrated'][T_metric][link_name] = {}
                evaluation_metrics[data_set]['uncalibrated'][link_name] = {}

    for link_name in active_link_dict:
        gamma_dict[link_name] = {}
        for link_value in active_link_dict[link_name]:
            if isinstance(link_value, tuple):
                # Round each element, then join with underscore
                key = "_".join(str(round(v, 2)) for v in link_value)
            else:
                # Just round the float
                key = str(round(link_value, 2))
            gamma_dict[link_name][key] = {}

    for link_name in active_link_dict:
        for link_value in active_link_dict[link_name]:

            if isinstance(link_value, tuple):
                # Round each element, then join with underscore
                key = "_".join(str(round(v, 2)) for v in link_value)
            else:
                # Just round the float
                key = str(round(link_value, 2))
            scaled_model = ModelWithTemperature(net, a=link_value, link=link_name)
            scaled_model.set_temperature(val_logits, val_labels, cross_validate='ece', device=device)
            T_opt_ce = scaled_model.get_temperature(metric='ce')
            T_opt_ece = scaled_model.get_temperature(metric='ece')
            
            gamma_dict[link_name][key]["CE"] = scaled_model.nll_vals
            gamma_dict[link_name][key]["ECE"]  = scaled_model.ece_vals

            ### First uncalibrated
            val_pred = get_probs(val_logits, T=1, a=link_value, link=link_name)
            evaluation_metrics['val']['uncalibrated'][link_name][key] = evaluate(val_labels, val_pred, num_classes=num_classes)
            test_pred = get_probs(test_logits, T=1, a=link_value, link=link_name)
            evaluation_metrics['test']['uncalibrated'][link_name][key] = evaluate(test_labels, test_pred, num_classes=num_classes)
                
            if train_labels is not None:
                train_pred = get_probs(train_logits, T=1, a=link_value, link=link_name)
                evaluation_metrics['train']['uncalibrated'][link_name][key] = evaluate(train_labels, train_pred, num_classes=num_classes)
            
                scaled_model.set_temperature(train_logits, train_labels, cross_validate='ece', device=device)

                for T_metric in ['ce', 'ece']:
                    train_T_opt = scaled_model.get_temperature(metric=T_metric)

                    train_pred = get_probs(train_logits, T=train_T_opt, a=link_value, link=link_name)
                    evaluation_metrics['train']['calibrated'][T_metric][link_name][key] = evaluate(train_labels, train_pred, num_classes=num_classes)

            for T_metric in ['ce', 'ece']:
                T_opt = T_opt_ce if T_metric == 'ce' else T_opt_ece
                val_pred = get_probs(val_logits, T=T_opt, a=link_value, link=link_name)
                evaluation_metrics['val']['calibrated'][T_metric][link_name][key] = evaluate(val_labels, val_pred, num_classes=num_classes)
                
                test_pred = get_probs(test_logits, T=T_opt, a=link_value, link=link_name)
                evaluation_metrics['test']['calibrated'][T_metric][link_name][key] = evaluate(test_labels, test_pred, num_classes=num_classes)
            
            for T_metric in ['ce', 'ece']:
                gamma_dict[link_name][key][" T_opt" + " " + T_metric] = T_opt_ce if T_metric == 'ce' else T_opt_ece
    
    evaluation_metrics['T_dict'] = gamma_dict
    return evaluation_metrics
