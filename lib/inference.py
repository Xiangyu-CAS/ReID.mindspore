import numpy as np
from mindspore.train.serialization import load_checkpoint

from lib.dataset.data_loader import get_test_loader
from lib.dataset.build_dataset import prepare_multiple_dataset
from lib.modeling.build_model import Encoder
from lib.evaluation import eval_func


def inference(cfg, logger):
    dataset = prepare_multiple_dataset(cfg, logger)
    test_loader = get_test_loader(dataset.query + dataset.gallery, cfg)
    model = Encoder(cfg)
    model.set_train(False)
    logger.info("loading model from {}".format(cfg.TEST.WEIGHT))
    load_checkpoint(cfg.TEST.WEIGHT, model)
    feats = []
    pids = []
    camids = []

    for batch in test_loader:
        data, pid, camid = batch
        feat = model(data)
        feats.append(feat.asnumpy())
        pids.extend(pid.asnumpy())
        camids.extend(camid.asnumpy())

    num_query = len(dataset.query)
    # query
    feats = np.concatenate(feats)
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
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return mAP
