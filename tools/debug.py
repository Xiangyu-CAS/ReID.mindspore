import sys
sys.path.append('.')
from lib.modeling.build_model import Encoder
from lib.config import _C as cfg


if __name__ == '__main__':
    config_file = './configs/naic20.yml'
    cfg.merge_from_file(config_file)
    cfg.freeze()
    model = Encoder(cfg)