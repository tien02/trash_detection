import os
import albumentations as A

# Data
ROOT = "../data"
TRAIN_DATA = "annotations_0_train.json"
VAL_DATA = "annotations_0_val.json"
TEST_DATA = "annotations_0_test.json"
BATCHSIZE = 4
NUM_WORKERS = 2

# Model BOUDING BOX TYPE
BBTYPE = 'xyxy' # cxcywh, xywh, xyxy
IMG_MIN_SIZE = 640
IMG_MAX_SIZE = 1080
NMS_INFERENCE = 0.3

# Model
NUM_CLASSES = 60
# NMS_THRESHOLD = 0.3
OPTIMIZER = 'Adam' # 'Adam', 'SGD', 'AdamW'
LR = 1e-3

# Trainer
ACCELERATOR = 'gpu'
NUM_EPOCHS = 300
EVAL_EVERY_EPOCH = 2

# Data augmentation
bbtype = ""
if BBTYPE == "xyxy":
    bbtype = "pascal_voc"
elif BBTYPE == "xywh":
    bbtype = "coco"
else:
    bbtype = "yolo"

transform = A.Compose([
    A.Resize(height=640, width=640, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    # A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format=bbtype, label_fields=['class_label']))

# TENSORBOARD LOG
TENSORBOARD = {
    "DIR": "FasterRCNN",
    "NAME": f"LABEL{NUM_CLASSES}_LR{LR}_OPT{OPTIMIZER}",
    "VERSION": "0",
}

# CHECKPOINT
CHECKPOINT_DIR = os.path.join(TENSORBOARD["DIR"], TENSORBOARD["NAME"], TENSORBOARD["VERSION"], "CKPT")

# Other
CONTINUE_TRAINING = None
TEST_CKPT_PATH = None