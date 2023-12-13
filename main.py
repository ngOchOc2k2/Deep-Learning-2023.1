import torch.optim as optim
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import MyDataset
from model.cross_encoder import DoubleBert
from train_v2 import MyTrainer
from config import * 
from torch.nn.parallel import DataParallel


    
        
if __name__ == '__main__':
    config = Configs


    data = MyDataset(config=config)

    model = DoubleBert(config=config)

    if config.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    
    train_data = DataLoader(dataset=data, batch_size=config.batch_size, num_workers=2, shuffle=False)

    trainer = MyTrainer(
        model=model,
        train_dataloader=train_data,
        epochs=config.epoch,
        lr=config.lr,
        save_step=config.save_step,
        log_step=config.log_step,
        freeze_encoder=False,
    )

    trainer.train()