import argparse
import os
import sys


sys.path.append('.')
from lib.config import _C as cfg
from lib.utils import setup_logger
from lib.dataset.build_dataset import prepare_multiple_dataset
from lib.train_net import Trainer, do_train


def naive_train(cfg, logger, distributed, local_rank):
    trainer = Trainer(cfg, distributed, local_rank)
    dataset = prepare_multiple_dataset(cfg, logger)
    # do_train(cfg, dataset)
    trainer.do_train(dataset)


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(cfg.LOGGER_NAME, output_dir, args.local_rank)

    if args.local_rank == 0:
        logger.info(args)
        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))

        logger.info("Running with config:\n{}".format(cfg))

    # TODO: distributed
    distributed = False
    naive_train(cfg, logger, distributed, args.local_rank)


if __name__ == '__main__':
    main()