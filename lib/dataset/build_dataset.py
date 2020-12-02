from .base import *

from .market1501 import Market1501

factory = {
    'market1501': Market1501,
}


def init_dataset(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)


def prepare_multiple_dataset(cfg, logger):
    dataset = BaseImageDataset()
    trainset = cfg.DATASETS.TRAIN
    valset = cfg.DATASETS.TEST
    for element in trainset:
        cur_dataset = init_dataset(element, root=cfg.DATASETS.ROOT_DIR)
        if cfg.DATASETS.COMBINEALL:
            dataset.train = merge_datasets([dataset.train, cur_dataset.train,
                                            cur_dataset.query + cur_dataset.gallery])
        else:
            dataset.train = merge_datasets([dataset.train, cur_dataset.train])
        dataset.relabel_train()

    if cfg.DATASETS.CUTOFF_LONGTAIL:
        dataset.train = cutoff_longtail(dataset.train, cfg.DATASETS.LONGTAIL_THR)
        dataset.relabel_train()

    for element in valset:
        cur_dataset = init_dataset(element, root=cfg.DATASETS.ROOT_DIR)
        lists = merge_datasets([dataset.query + dataset.gallery, cur_dataset.query + cur_dataset.gallery])
        query = lists[:len(dataset.query)] + \
                lists[len(dataset.query + dataset.gallery): len(dataset.query + dataset.gallery) + len(cur_dataset.query)]
        gallery = lists[len(dataset.query):len(dataset.query + dataset.gallery)]+ \
                  lists[len(dataset.query + dataset.gallery) + len(cur_dataset.query):]
        dataset.query = query
        dataset.gallery = gallery

    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery, logger)
    return dataset


def cutoff_longtail(dataset, thr):
    labels = {}
    for img_path, pid, camid in dataset:
        if pid in labels:
            labels[pid].append([img_path, pid, camid])
        else:
            labels[pid] = [[img_path, pid, camid]]
    keep_data = []
    remove_data = []
    for key, value in labels.items():
        if len(value) < thr:
            remove_data.extend(value)
            continue
        keep_data.extend(value)
    return keep_data


# exchange pid and camid, used in training camera bias model
def exchange_pid_camid(dataset):
    train = []
    query = []
    gallery = []
    for img_path, pid, camid in dataset.train:
        train.append([img_path, camid, pid])
    for img_path, pid, camid in dataset.query:
        query.append([img_path, camid, pid])
    for img_path, pid, camid in dataset.gallery:
        gallery.append([img_path, camid, pid])
    dataset.train = train
    dataset.query = query
    dataset.gallery = gallery
    dataset.relabel_train()
    return dataset