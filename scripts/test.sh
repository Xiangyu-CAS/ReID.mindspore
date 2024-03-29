python ./tools/test.py \
--config_file='configs/naic20.yml' \
MODEL.BACKBONE 'resnet50' \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/xiangyuzhu/data/ReID' \
DATASETS.TRAIN "('market1501',)" \
DATASETS.TEST "('market1501',)" \
TEST.WEIGHT './output/debug/final.ckpt'

# TEST.WEIGHT 'torch2ms/encoder-bn2d.ckpt'