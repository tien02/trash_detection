import config
from utils import apply_nms

from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim import Adam, SGD, AdamW

class FasterRCNNTrainer(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.test_metrics = MeanAveragePrecision()
        self.val_metrics = MeanAveragePrecision()

    def forward(self, imgs):
        pred = self.model(imgs)
        return pred

    def training_step(self, batch):
        imgs, boxes, labels = batch
        targets = {
            'boxes': boxes,
            'labels': labels
        }
        
        losses = self.model(imgs, targets)
        sum_loss = sum(losses.values())
        self.log("train_loss", sum_loss, on_epoch=True, prog_bar=True)
        return sum_loss
    
    def test_step(self, batch, batch_idx):
        imgs, boxes, labels = batch

        pred = self.model(imgs)
        pred_after_nms = apply_nms(pred, config.NMS_THRESHOLD)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        self.test_metrics.update(pred_after_nms, target) 

    def validation_step(self, batch, batch_idx):
        imgs, boxes, labels = batch

        pred = self.model(imgs)
        pred_after_nms = apply_nms(pred, config.NMS_THRESHOLD)

        target = {
            'boxes': boxes,
            'labels': labels
        }
        self.val_metrics.update(pred_after_nms, target)
    
    def validation_epoch_end(self, outputs):
        metrics = self.val_metrics.compute()
        self.log("val_metrics", metrics, prog_bar=True)
        self.val_metrics.reset()
    
    def configure_optimizers(self):
        ## Optimizer
        if config.OPTIMIZER == "Adam":
            optimizer = Adam(self.model.parameters(), lr=config.LR, weight_decay=0.01)
        elif config.OPTIMIZER == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=config.LR, weight_decay=0.01)
        else:
            optimizer = SGD(self.model.parameters(), lr=config.LR, weight_decay=0.01)
            
        ## Scheduler
        # scheduler = 
            
            return {
                'optimizer': optimizer
                # 'scheduler':
            }