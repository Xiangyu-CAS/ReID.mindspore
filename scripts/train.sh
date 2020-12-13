python ./tools/train.py \
--config_file='configs/naic20.yml' \
MODEL.BACKBONE 'resnet50' \
MODEL.PRETRAIN_CHOICE 'imagenet' \
MODEL.PRETRAIN_PATH './torch2ms/resnet50-torch.ckpt' \
DATASETS.ROOT_DIR '/home/xiangyuzhu/data/ReID' \
DATASETS.TRAIN "('market1501',)" \
DATASETS.TEST "('market1501',)" \
SOLVER.MAX_EPOCHS 30 \
SOLVER.BATCH_SIZE 128 \
SOLVER.BASE_LR 7e-4 \
SOLVER.EVAL_PERIOD 100 \
OUTPUT_DIR './output/debug'
