import os
import cv2
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.py_transforms as T
import mindspore.dataset.transforms.py_transforms as T2

import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

from PIL import Image

from lib.dataset.transforms.augmix import AugMix


class ImageDataset():
    def __init__(self, dataset, transform=None):
        super(ImageDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid


def get_train_loader(dataset, cfg):
    height, width = cfg.INPUT.SIZE
    normalizer = T.Normalize(mean=cfg.INPUT.MEAN,
                             std=cfg.INPUT.STD)

    train_transformer = T2.Compose([
        T.ToPIL(),
        T.Resize((height, width)),
        T.RandomHorizontalFlip(prob=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T2.RandomApply([T.RandomColorAdjust(brightness=cfg.INPUT.CJ_PARAM[0], contrast=cfg.INPUT.CJ_PARAM[1],
                                            saturation=cfg.INPUT.CJ_PARAM[2], hue=cfg.INPUT.CJ_PARAM[3])],
                       prob=cfg.INPUT.CJ_PROB),
        AugMix(prob=cfg.INPUT.AUGMIX_PROB), # augmix: PIL to numpy
        T.ToTensor(),
        T.RandomErasing(prob=cfg.INPUT.RE_PROB),
        normalizer,
    ])

    # TODO: PK sampler
    sampler = None

    dataset = ImageDataset(dataset, train_transformer)
    de_dataset = de.GeneratorDataset(dataset, ["image", "pid", "camid"],
                                     num_parallel_workers=cfg.DATALOADER.NUM_WORKERS,
                                     sampler=sampler,
                                     shuffle=True)
    type_cast_op = C2.TypeCast(mstype.int32)
    de_dataset = de_dataset.map(operations=type_cast_op, input_columns="pid")
    de_dataset = de_dataset.batch(batch_size=cfg.SOLVER.BATCH_SIZE, drop_remainder=True)
    return de_dataset


def get_test_loader(dataset, cfg):
    height, width = cfg.INPUT.SIZE
    normalizer = T.Normalize(mean=cfg.INPUT.MEAN,
                             std=cfg.INPUT.STD)
    test_transformer =T2.Compose([
        T.ToPIL(),
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])
    dataset = ImageDataset(dataset, test_transformer)
    de_dataset = de.GeneratorDataset(dataset, ["image", "pid", "camid"],
                                     num_parallel_workers=cfg.DATALOADER.NUM_WORKERS,
                                     sampler=None,
                                     shuffle=False)
    de_dataset = de_dataset.batch(batch_size=128, drop_remainder=True)
    return de_dataset
