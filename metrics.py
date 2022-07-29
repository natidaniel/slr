"""
Performance metric functions
"""

import numpy as np
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.preprocessing import label_binarize


def cluster_acc(y_true, y_pred):
    """
    Calculate unsupervised clustering accuracy. Requires scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return metrics.accuracy_score(y_pred_voted, y_true)


def roc_auc(y_true, q_pred, n_classes):
    """
    Calculate area under ROC curve (ROC AUC)

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        q_pred: predicted probabilities, numpy.array with shape `(n_samples,)`

    # Return
        ROC AUC score, in [0,1]
    """
    if n_classes == 2:  # binary ROC AUC
        auc = max(metrics.roc_auc_score(y_true, q_pred[:, 1]), metrics.roc_auc_score(y_true, q_pred[:, 0]))
    else:  # micro-averaged ROC AUC (multiclass)
        fpr, tpr, _ = metrics.roc_curve(label_binarize(y_true, classes=np.unique(y_true)).ravel(), q_pred.ravel())
        auc = metrics.auc(fpr, tpr)
    return auc
