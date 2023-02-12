import albumentations as A

# Model
NUM_CLASSES = 60

# Trainer
NMS_THRESHOLD = 0.3
OPTIMIZER = 'Adam' # 'Adam', 'SGD', 'AdamW'
LR = 1e-3

# Data augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_label']))