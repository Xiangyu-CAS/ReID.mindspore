import time
import os
import numpy as np
import logging

import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.nn.optim import Adam, SGD
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import save_checkpoint
from mindspore import context

from lib.dataset.build_dataset import prepare_multiple_dataset
from lib.modeling.build_model import *
from lib.dataset.data_loader import get_train_loader, get_test_loader
from lib.utils import *
from lib.evaluation import eval_func
from lib.losses.build_loss import build_loss_fn, CrossEntropySmooth
from lib.solver import *

'''
high level api for training, recommend by official
'''
def do_train(cfg, dataset):
    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="GPU")

    num_class, _, _ = dataset.get_imagedata_info(dataset.train)

    # dataset
    train_loader = get_train_loader(dataset.train, cfg)
    test_loader = get_test_loader(dataset.query + dataset.gallery, cfg)

    # scheduler
    scheduler = warmup_cosine_annealing_lr(cfg.SOLVER.BASE_LR, train_loader.get_batch_size(),
                                           warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
                                           max_epoch=cfg.SOLVER.MAX_EPOCHS)
    # model
    encoder = Encoder(cfg)
    model = Head(encoder, num_class, cfg)
    #model =resnet50(num_class)

    # optimizer
    params = model.trainable_params()
    optimizer = Adam(params, scheduler, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # loss_fn
    loss_fn = CrossEntropySmooth(num_classes=num_class)

    loss_scale = FixedLossScaleManager(256, drop_overflow_update=False)
    # Mixed precision
    model = Model(model, loss_fn=loss_fn, optimizer=optimizer, loss_scale_manager=loss_scale, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=True)

    # callbacks
    loss_cb = LossMonitor()
    model.train(cfg.SOLVER.MAX_EPOCHS, train_loader, callbacks=[loss_cb])

    return


class Trainer(object):
    def __init__(self, cfg, distributed=False, local_rank=0):
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target="GPU")

        self.encoder = Encoder(cfg)
        self.cfg = cfg

        self.distributed = distributed
        self.local_rank = local_rank

        self.logger = logging.getLogger("{}.train".format(cfg.LOGGER_NAME))
        self.best_mAP = 0

    def do_train(self, dataset):
        num_class, _, _ = dataset.get_imagedata_info(dataset.train)

        # dataset
        train_loader = get_train_loader(dataset.train, self.cfg)
        test_loader = get_test_loader(dataset.query + dataset.gallery, self.cfg)

        # scheduler
        scheduler = warmup_cosine_annealing_lr(self.cfg.SOLVER.BASE_LR, train_loader.get_batch_size(),
                                               warmup_epochs=self.cfg.SOLVER.WARMUP_EPOCHS,
                                               max_epoch=self.cfg.SOLVER.MAX_EPOCHS)
        # model
        model = Head(self.encoder, num_class, self.cfg)

        # optimizer
        params = model.trainable_params()
        if self.cfg.SOLVER.OPTIMIZER == 'Adam':
            optimizer = Adam(params, scheduler, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.OPTIMIZER == 'SGD':
            optimizer = SGD(params, scheduler, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
                            momentum=0.9)
        else:
            raise RuntimeError("unknown optimizer: '{}'".format(self.cfg.SOLVER.OPTIMIZER))

        # loss
        loss_fn = CrossEntropySmooth(num_classes=num_class) #build_loss_fn(self.cfg, num_class)
        model = NetworkWithLoss(model, loss_fn)
        train_net = TrainWrapper(model, optimizer)

        # train
        for epoch in range(self.cfg.SOLVER.MAX_EPOCHS):
            if self.local_rank == 0:
                self.logger.info("Epoch[{}]"
                                 .format(epoch))
            self.train_epoch(train_net, train_loader, epoch)

            # validation
            if self.local_rank == 0 and (epoch % self.cfg.SOLVER.EVAL_PERIOD == 0 or epoch == self.cfg.SOLVER.MAX_EPOCHS - 1):
                cur_mAP = self.validate(test_loader, len(dataset.query))
                if cur_mAP >= self.best_mAP:
                    self.best_mAP = cur_mAP
                    save_checkpoint(self.encoder, os.path.join(self.cfg.OUTPUT_DIR, 'best.pth'))
        self.logger.info("best mAP: {:.1%}".format(self.best_mAP))

    def train_epoch(self, train_net, train_loader, epoch):
        train_net.set_train(True)
        id_losses = AverageMeter()
        metric_losses = AverageMeter()
        data_time = AverageMeter()
        model_time = AverageMeter()

        # TODO: freeze first epoch

        start = time.time()
        data_start = time.time()
        for iteration, batch in enumerate(train_loader):
            input, target, camids = batch
            input = Tensor(input)
            target = Tensor(target, mstype.int32)
            data_time.update(time.time() - data_start)

            model_start = time.time()
            loss = train_net(input, target)

            id_losses.update(loss.asnumpy(), input.size())
            metric_losses.update(loss.asnumpy(), input.size())

            model_time.update(time.time() - model_start)
            data_start = time.time()

            if iteration % 100 == 0 and self.local_rank == 0:
                self.logger.info("Epoch[{}] Iteration[{}/{}] ID Loss: {:.3f}, "
                                 "Metric Loss: {:.3f}, data time: {:.3f}s, model time: {:.3f}s"
                        .format(epoch, iteration, train_loader.get_dataset_size(),
                                id_losses.val, metric_losses.val, data_time.val, model_time.val))
        train_loader.reset()
        end = time.time()
        if self.local_rank == 0:
            self.logger.info("epoch takes {:.3f}s".format((end - start)))

    def validate(self, test_loader, num_query):
        self.encoder.set_train(False)
        feats = []
        pids = []
        camids = []

        for batch in test_loader:
            data, pid, camid = batch
            data = Tensor(data)
            feat = self.encoder(data)

            feats.append(feat)
            pids.extend(pid.asnumpy())
            camids.extend(camid.asnumpy())

        feats = P.Concat(axis=0)(tuple(feats))
        test_loader.reset()

        # query
        feats = feats.asnumpy()
        qf = feats[:num_query, ]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats[num_query:, ]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])

        # numpy
        sim = np.matmul(qf, gf.transpose())
        indices = np.argsort(-sim, axis=1)

        cmc, mAP = eval_func(indices, q_pids, g_pids, q_camids, g_camids)
        self.logger.info("Validation Results")
        self.logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return mAP