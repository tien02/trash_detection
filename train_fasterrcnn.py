import config
from dataset import TacoDataset
from model import get_fasterrcnn
from trainer import FasterRCNNTrainer
from utils import collate_fn

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == '__main__':
    # Seed
    seed_everything(42)

    # Data
    train_data = TacoDataset(root=config.ROOT, 
                            annFile=config.TRAIN_DATA, 
                            bbformat='xyxy', 
                            transform=config.transform)
    
    val_data = TacoDataset(root=config.ROOT, 
                            annFile=config.VAL_DATA, 
                            bbformat='xyxy')

    # DataLoader
    train_dataloader = DataLoader(train_data, 
                                batch_size=config.BATCHSIZE, 
                                shuffle=True, 
                                num_workers=config.NUM_WORKERS,
                                collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, 
                                batch_size=config.BATCHSIZE, 
                                num_workers=config.NUM_WORKERS,
                                collate_fn=collate_fn)

    # Model
    fasterrcnn = get_fasterrcnn(out_features=config.NUM_CLASSES)
    model = FasterRCNNTrainer(model=fasterrcnn)

    # Callback
    ckpt = ModelCheckpoint(dirpath=config.CHECKPOINT_DIR,monitor='map', save_top_k=2, mode='max')
    early_stopping = EarlyStopping(monitor="map", mode='max', check_finite=True)
    tensorboard = TensorBoardLogger(save_dir=config.TENSORBOARD["DIR"], 
                                    name=config.TENSORBOARD["NAME"], 
                                    version=config.TENSORBOARD["VERSION"])

    # Trainer
    trainer = Trainer(accelerator=config.ACCELERATOR, 
                    check_val_every_n_epoch=config.EVAL_EVERY_EPOCH,
                    gradient_clip_val=1.0, 
                    max_epochs=config.NUM_EPOCHS, 
                    enable_checkpointing=True, 
                    deterministic=True,
                    default_root_dir=config.CHECKPOINT_DIR, 
                    callbacks=[ckpt, early_stopping, tensorboard])

    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path = config.CONTINUE_TRAINING) 

    print(" == Finish Training == ")               