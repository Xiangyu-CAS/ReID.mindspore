import argparse
import os
import sys


sys.path.append('.')
from lib.config import _C as cfg
from lib.utils import setup_logger
from lib.inference import inference


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

    output_dir = os.path.dirname(cfg.TEST.WEIGHT)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(cfg.LOGGER_NAME, output_dir, args.local_rank, file_name='log_test.txt')
    inference(cfg, logger)


if __name__ == '__main__':
    main()