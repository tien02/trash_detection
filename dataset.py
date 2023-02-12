import os
import numpy as np

import torch
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection 

from utils import xywh2cxcywh, xywh2xyxy

class TacoDataset(CocoDetection):
    def __init__(self, root, annFile, bbformat='xyxy', transform=None):
        annFile = os.path.join(root, annFile)
        super().__init__(root, annFile)
        self.transform = transform
        self.bbformat = bbformat
    
    def __getitem__(self, idx):
        img, label = super(TacoDataset, self).__getitem__(idx)
        box = label[0]["bbox"]
        category = label[0]['category_id']
        
        if self.bbformat == 'xyxy':
            box = xywh2xyxy(box)

        if self.bbformat == 'cxcywh':
            box = xywh2cxcywh(box)        

        if self.transform:
            sample = {
                'image':np.array(img),
                'bboxes': [box],
                'class_label': [category]
            }

            transformed = self.transform(**sample)
            img, box, category = transformed.values()
        target = {
            'boxes': torch.tensor(box),
            'labels': torch.tensor(category)
        }
        return transforms.ToTensor()(img), target