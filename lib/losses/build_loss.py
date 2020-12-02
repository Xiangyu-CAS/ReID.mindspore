import mindspore.nn as nn
from mindspore.ops import functional as F

from .id_loss import *


def build_loss_fn(cfg, num_classes):
    if cfg.MODEL.ID_LOSS_TYPE == 'none':
        def id_loss_fn(score, target):
            return 0
    else:
        if cfg.MODEL.SMOOTH_LABEL:
            id_loss_fn = CrossEntropySmooth(num_classes=num_classes)
        else:
            id_loss_fn = nn.SoftmaxCrossEntropyWithLogits()

    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        metric_loss_fn = TripletLoss(margin=cfg.MODEL.TRIPLET_MARGIN, scale=cfg.MODEL.METRIC_LOSS_SCALE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'circle':
        metric_loss_fn = CircleLoss(m=cfg.MODEL.METRIC_LOSS_MARGIN, s=cfg.MODEL.METRIC_LOSS_SCALE)
    else:
        def metric_loss_fn(feat, target, feat_t, target_t):
            return 0

    def loss_func(score, feat, target, feat_t, target_t):
        return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_fn(score, target), \
               cfg.MODEL.METRIC_LOSS_WEIGHT * metric_loss_fn(feat, target, feat_t, target_t)

    return loss_func
