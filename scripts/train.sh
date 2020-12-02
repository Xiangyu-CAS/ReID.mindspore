python ./tools/train.py \
--config_file='configs/naic20.yml' \
MODEL.BACKBONE 'resnet50' \
MODEL.PRETRAIN_CHOICE 'scratch' \
DATASETS.ROOT_DIR '/home/xiangyuzhu/data/ReID' \
DATASETS.TRAIN "('market1501',)" \
DATASETS.TEST "('market1501',)" \
OUTPUT_DIR './output/debug'
