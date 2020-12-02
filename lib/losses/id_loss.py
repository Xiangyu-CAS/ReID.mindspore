import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class CrossEntropySmooth(_Loss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0.1, num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss