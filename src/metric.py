import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def get_auc(y_score, y_true):
    # for Validation sanity check:
    if y_true.shape[0] == 1 or y_true.shape[0] == 2:
        return 0, 0
    else:
        roc_aucs = roc_auc_score(y_true.flatten(0, 1).detach().cpu().numpy(),y_score.flatten(0, 1).detach().cpu().numpy())
        pr_aucs = average_precision_score(y_true.flatten(0, 1).detach().cpu().numpy(),y_score.flatten(0, 1).detach().cpu().numpy())
        return roc_aucs, pr_aucs
