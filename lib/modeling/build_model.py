import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint

from .resnet import resnet50, resnet101


backbone_factory = {
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


def build_backbone(name, *args, **kwargs):
    if name not in backbone_factory.keys():
        raise KeyError("Unknown backbone: {}".format(name))
    return backbone_factory[name][0](*args, **kwargs), backbone_factory[name][1]


class Encoder(nn.Cell):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.base, self.in_planes = build_backbone(cfg.MODEL.BACKBONE)

        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            load_checkpoint(cfg.MODEL.PRETRAIN_PATH, net=self.base)
            print('Loading pretrained ImageNet model......')
        elif cfg.MODEL.PRETRAIN_CHOICE == 'scratch':
            print('training from scratch....')

        # TODO： GeM
        self.gap = P.ReduceMean(keep_dims=True)

        self.flatten = nn.Flatten()
        # BatchNorm1d is only supported on ascend
        # self.bottleneck = nn.SequentialCell([nn.Dense(self.in_planes, cfg.MODEL.REDUCE_DIM),
        #                                      nn.BatchNorm1d(cfg.MODEL.REDUCE_DIM),
        #                                      ])
        self.bottleneck = nn.SequentialCell([nn.Dense(self.in_planes, cfg.MODEL.REDUCE_DIM),
                                             ])
        self.in_planes = cfg.MODEL.REDUCE_DIM

    def construct(self, x):
        featmap = self.base(x)
        global_feat = self.gap(featmap, (2, 3))
        global_feat = self.flatten(global_feat)
        feat = self.bottleneck(global_feat)

        # L2 norm is only supported on ascend
        # feat = P.L2Normalize(feat)
        return feat


class Head(nn.Cell):
    def __init__(self, encoder, num_class, cfg):
        super(Head, self).__init__()
        self.encoder = encoder
        self.id_loss_type = cfg.MODEL.ID_LOSS_TYPE

        # TODO: circle loss
        self.classifier = nn.Dense(self.encoder.in_planes, num_class, has_bias=False)

    def construct(self, x, label=None):
        feat = self.encoder(x)
        score = self.classifier(feat)

        return score, feat


class NetworkWithLoss(nn.Cell):
    def __init__(self, model, criterion):
        super(NetworkWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def construct(self, input_data, target):
        score, feat = self.model(input_data)
        id_loss, metric_loss = self.criterion(score, feat, target, None, None)
        loss = id_loss + metric_loss
        return loss
