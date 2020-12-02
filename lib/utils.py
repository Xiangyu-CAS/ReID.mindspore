import cv2
import os
import sys
import logging
import json
import numpy as np

import mindspore as ms


def setup_logger(name, save_dir, distributed_rank, file_name='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, file_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_files(dir_path, img_paths, suffix=['jpg', 'png', 'bmp']):
    files = os.listdir(dir_path)
    for f in files:
        path = os.path.join(dir_path, f)
        if os.path.isdir(path):
            find_files(path, img_paths, suffix)
        elif f.split('.')[-1] in suffix:
            img_paths.append(path)


def write_json(indices, q_img_paths, g_img_paths, dst_dir,
               name='submit.json', top_k=200):
    results = {}
    indices = indices[:, :top_k]
    for i in range(indices.shape[0]):
        query_name = os.path.basename(q_img_paths[i])
        results[query_name] = []
        for j in range(indices.shape[1]):
            idx = indices[i, j]
            gallery_name = os.path.basename(g_img_paths[idx])
            results[query_name].append(gallery_name)
    with open(os.path.join(dst_dir, name), 'w', encoding='utf-8') as f:
        json.dump(results, f)


def vis_rank(dataset, indices, out_dir, topk=5, size=(128, 256)):
    out_dir = os.path.join(out_dir, 'vis_rank')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    color = (0, 0, 255)
    indices = indices[:, :topk]
    img_paths = []
    pids = []
    camids = []
    for img_path, pid, camid in dataset.query + dataset.gallery:
        img_paths.append(img_path)
        pids.append(pid)
        camids.append(camid)
    num_query = len(dataset.query)
    for i in range(indices.shape[0]):
        query_img = cv2.imread(img_paths[i])
        query_img = cv2.resize(query_img, size)
        imgs = [query_img]
        for j in range(indices.shape[1]):
            idx = num_query + indices[i, j]
            img = cv2.imread(img_paths[idx])
            img = cv2.resize(img, size)
            if pids[i] != pids[idx]:
                img = cv2.rectangle(img, (0, 0), size, color, 4)
            imgs.append(img)
        canvas = np.concatenate(imgs, axis=1)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_paths[i])), canvas)
