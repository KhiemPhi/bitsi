"""Segmentation metrics (Sec. 5.1)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def class_agnostic_miou(gt, pred):
    """Hungarian-matched mean IoU over ground-truth parts of one object.

    gt, pred: (N,) integer labels. Unmatched ground-truth parts count as IoU 0.
    """
    g_ids, g = np.unique(gt, return_inverse=True)
    p_ids, p = np.unique(pred, return_inverse=True)
    inter = np.zeros((len(g_ids), len(p_ids)))
    np.add.at(inter, (g, p), 1)
    gs = inter.sum(1, keepdims=True)
    ps = inter.sum(0, keepdims=True)
    iou = inter / (gs + ps - inter)
    r, c = linear_sum_assignment(-iou)
    per = np.zeros(len(g_ids))
    per[r] = iou[r, c]
    return float(per.mean()), dict(zip(g_ids.tolist(), per.tolist()))


def partition_scores(gt, pred):
    """Scores of a flat partition that also see over-segmentation.

    miou      Hungarian mIoU averaged over GT parts (the metric used so far; extra predicted
              segments cost nothing directly)
    sym_miou  the same matched IoUs averaged over max(#GT, #pred): every unmatched predicted
              segment counts as a 0, so over- and under-segmentation are both paid for
    ari       adjusted Rand index of the two partitions (pair counting; 0 = chance, 1 = equal)
    n_pred, n_gt  segment counts
    """
    from sklearn.metrics import adjusted_rand_score
    g_ids, g = np.unique(gt, return_inverse=True)
    p_ids, p = np.unique(pred, return_inverse=True)
    inter = np.zeros((len(g_ids), len(p_ids)))
    np.add.at(inter, (g, p), 1)
    iou = inter / (inter.sum(1, keepdims=True) + inter.sum(0, keepdims=True) - inter)
    r, c = linear_sum_assignment(-iou)
    matched = iou[r, c].sum()
    return dict(miou=float(matched / len(g_ids)), sym_miou=float(matched / max(len(g_ids), len(p_ids))),
                ari=float(adjusted_rand_score(g, p)), n_pred=int(len(p_ids)), n_gt=int(len(g_ids)))


def node_gt_iou(hier, sp_labels, gt):
    """IoU of every hierarchy node with every ground-truth part.

    Returns (n_nodes, G) IoU matrix and the GT ids.
    """
    g_ids, g = np.unique(gt, return_inverse=True)
    M = hier.n_leaves
    cnt = np.zeros((M, len(g_ids)))
    np.add.at(cnt, (sp_labels, g), 1)
    node_cnt = np.zeros((hier.n_nodes, len(g_ids)))
    node_cnt[:M] = cnt
    for a, b, new, _ in hier.merges:
        node_cnt[new] = node_cnt[a] + node_cnt[b]
    node_size = node_cnt.sum(1, keepdims=True)
    gt_size = cnt.sum(0, keepdims=True)
    iou = node_cnt / (node_size + gt_size - node_cnt)
    return iou, g_ids


def hierarchy_recall(hier, sp_labels, gt, thresh=0.5):
    """Fraction of GT parts matched by some node of H with IoU >= thresh, and best IoUs."""
    iou, g_ids = node_gt_iou(hier, sp_labels, gt)
    best = iou.max(0)
    return float((best >= thresh).mean()), dict(zip(g_ids.tolist(), best.tolist()))


def superpoint_purity(sp_labels, gt):
    """Upper bound on any segmentation built from these superpoints: fraction of points
    whose superpoint majority label equals their own label."""
    M = sp_labels.max() + 1
    g_ids, g = np.unique(gt, return_inverse=True)
    cnt = np.zeros((M, len(g_ids)))
    np.add.at(cnt, (sp_labels, g), 1)
    return float(cnt.max(1).sum() / len(gt))
