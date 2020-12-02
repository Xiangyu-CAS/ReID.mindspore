from yacs.config import CfgNode as CN

_C = CN()
_C.LOGGER_NAME = 'ReID.mindspore'
# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
_C.DEVICE = 'gpu' # {gpu, ascend}

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.BACKBONE = 'resnet50_ibn_a'
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# syncBN
_C.MODEL.USE_SYNCBN = False
_C.MODEL.POOLING = 'GeM' #{GeM, avg}
_C.MODEL.REDUCE = False
_C.MODEL.REDUCE_DIM = 512

_C.MODEL.MEMORY_BANK = False
_C.MODEL.MEMORY_SIZE = 8192 #4096, 8192, 16384, 32768
_C.MODEL.EMA_MOMENTUM = 0.999

_C.MODEL.ID_LOSS_TYPE = 'softmax' #{softmax, Cosface, Circle}
_C.MODEL.METRIC_LOSS_TYPE = 'none' #{}

# parameters for large margin softmax loss (eg. arcface, cosface, circle loss)
_C.MODEL.ID_LOSS_MARGIN = 0.35
_C.MODEL.ID_LOSS_SCALE = 30

# paramters for large margin metric loss (eg. circle loss)
_C.MODEL.METRIC_LOSS_MARGIN = 0.25
_C.MODEL.METRIC_LOSS_SCALE = 96

## margin for triplet loss
_C.MODEL.TRIPLET_MARGIN = 0.0

_C.MODEL.ID_LOSS_WEIGHT = 1.
_C.MODEL.METRIC_LOSS_WEIGHT = 1.

_C.MODEL.SMOOTH_LABEL = True
# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE = [256, 128] # input size
_C.INPUT.AUGMIX_PROB = 0.0 # augmix
_C.INPUT.AUTOAUG_PROB = 0.0 # auto augmentation
_C.INPUT.CJ_PROB = 0.0 # Color jitter
_C.INPUT.CJ_PARAM = [0.15, 0.1, 0., 0.]
_C.INPUT.RE_PROB = 0.0#random erase
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.FP16 = False
_C.SOLVER.BATCH_SIZE = 64
# steps for gradient accumulation
_C.SOLVER.ACCUMULATE_STEPS = 1
_C.SOLVER.MAX_EPOCHS = 20
_C.SOLVER.BASE_LR = 3.5e-4
_C.SOLVER.WEIGHT_DECAY = 5e-4

_C.SOLVER.FREEZE_EPOCHS = 0

_C.SOLVER.WARMUP_EPOCHS = 4

_C.SOLVER.EVAL_PERIOD = 2

_C.SOLVER.OPTIMIZER = 'Adam'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'random' #{random, randomIdentity}
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = './data'

_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()
# Combine train, query, gallery of trainset
_C.DATASETS.COMBINEALL = False

_C.DATASETS.CUTOFF_LONGTAIL = False
_C.DATASETS.LONGTAIL_THR = 4
#-------------------------------------------------------------------------------
# Test
#-------------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.WEIGHT = ''
_C.TEST.FLIP_TEST = False
# database augmentation
_C.TEST.DO_DBA = False
# query expansion
_C.TEST.DO_QE = False
# re-rank
_C.TEST.DO_RERANK = False
_C.TEST.RERANK_PARAM = [20, 6, 0.3]

_C.TEST.WRITE_FEAT = False
_C.TEST.WRITE_RESULTS = False
_C.TEST.CAM_DISTMAT = ''

_C.TEST.VIS_RANK = False
_C.OUTPUT_DIR = ''