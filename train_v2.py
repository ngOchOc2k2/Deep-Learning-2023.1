import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import MyDataset
from tqdm import tqdm
import logging
from config import *
from loss import ContrastiveLoss, CrossEntropyLoss, CombineLoss
import json


logger = logging.getLogger(__name__)

config = Configs


class MyTrainer():
    def __init__(self, 
                 model=None, 
                 train_dataloader=None, 
                 epochs=None, 
                 scheduler=None, 
                 lr=1e-5, 
                 save_step=100, 
                 use_amp=True,
                 log_step=100, 
                 weight_decay=0.1, 
                 evaluation_steps=None, 
                 save_path=None, 
                 save_best_model=None, 
                 max_grad_norm=None, 
                 show_progress_bar=True, 
                 device=None,
                 freeze_encoder=None):
        
        self.model = model
        self.tokenizer = model
        
        self.train_dataloader = train_dataloader
        self.epochs = epochs

        self.scheduler = scheduler
        self.lr = lr
        
        self.weight_decay = weight_decay
        self.evaluation_steps = evaluation_steps
        
        self.save_path = save_path
        self.save_best_model = save_best_model
        
        self.max_grad_norm = max_grad_norm
        self.show_progress_bar = show_progress_bar
        
        self.save_step = save_step
        self.log_step = log_step
        
        self.activation_fct = nn.GELU()
        self.freeze_encoder = freeze_encoder
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)


        self.best_score = -9999999
        self.num_train_steps = int(len(train_dataloader) * epochs)
        self.loss = CombineLoss()
        self.loss_2 = CrossEntropyLoss()

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        criterion = self.loss
        # criterion_2 = self.loss_2
        
        trainable_params_count = 0

        if self.freeze_encoder == 'only_mlp':
            no_decay = []
            cls_bias = []
            cls_weight = []
            layers = ["linear_layer.weight", "linear_layer.bias", "normalize.weight", "normalize.bias", "last_output.weight", "last_output.bias"]

            layers.extend(cls_weight)
            no_decay.extend(cls_bias)   

            optimizer = [
                
                {
                    "params": [p for n, p in self.model.named_parameters() if any([nd in n for nd in layers])],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        elif self.freeze_encoder == 'full':
            no_decay = []
            cls_bias = []
            cls_weight = []
            layers = ["linear_layer.weight", "linear_layer.bias", "normalize.weight", "normalize.bias", "last_output.weight", "last_output.bias"]

            layers.extend(cls_weight)
            no_decay.extend(cls_bias)   

            optimizer = [
                
                {
                    "params": [p for n, p in self.model.named_parameters() if not any([nd in n for nd in no_decay])],
                    "weight_decay": 0.1,
                },
            ]

        else:
            no_decay = ["adapter.proj_up.bias", "adapter.proj_down.bias", "LayerNorm.bias"]
            classifier = ["linear_layer.weight", "linear_layer.bias", "normalize.weight", "normalize.bias", "last_output.weight", "last_output.bias"]
            layers = ["adapter.proj_up.weight", "adapter.proj_down.weight", "LayerNorm.weight"]
            
            layers.extend(classifier)
            # no_decay.extend(cls_bias)   
            
            optimizer = [
                
                {
                    "params": [p for n, p in self.model.named_parameters() if any([nd in n for nd in layers])],
                    "weight_decay": 0.1,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },

            ]


        for group in optimizer:
            for param in group["params"]:
                trainable_params_count += param.numel()
        print(f'Total Trainable params: {trainable_params_count}')
        optimizer = optim.Adam(optimizer, lr=self.lr)
        
        list_loss = []

        logger.warning('---' * 16 + 'Starting training' + '---' * 16)
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            total_loss = 0.0
            
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                query = batch["query_text"]
                positive = batch["positive_text"]
                negative = batch["negative_text"]

                
                optimizer.zero_grad()

                list_negative = []
                list_soft_neg = []
                
                for neg in negative:
                    sim_neg, soft_neg = self.model(query, neg)
                    list_negative.append(sim_neg)
                    list_soft_neg.append(soft_neg)
                    
                sim_pos, soft_pos = self.model(query, positive)
                
                
                # print(len(query))
                label_pos = [1 for _ in range(len(query))]
                label_neg = [0 for _ in range(len(query) * 5)]
                
                label_positive = torch.tensor(label_pos, dtype=torch.long).to(config.device)
                label_negative = torch.tensor(label_neg, dtype=torch.long).to(config.device)
                
                sim_neg = torch.cat(list_negative)
                sim_neg = sim_neg.unsqueeze(1)

                soft_neg = torch.cat(list_soft_neg)
                soft_neg = soft_neg.unsqueeze(1)
                
                loss = criterion(
                    softmax_positive=soft_pos,
                    softmax_negative=soft_neg,
                    label_positive=label_positive,
                    label_negative=label_negative,
                    positive_similarities=sim_pos,
                    negative_similarities=sim_neg,
                    device=config.device
                )
                
                # print(f"Label pos: {label_pos}")
                # print(f"Label neg: {label_neg}")
                
                # print(f"Similar neg: {soft_neg}")
                # print(f"Similar pos: {soft_pos}")
                
                
                # loss_2 = criterion_2(
                #     softmax_positive=soft_pos,
                #     softmax_negative=soft_neg,
                #     label_positive=label_pos,
                #     label_negative=label_neg,
                # )


                loss.backward()
                optimizer.step()
                total_loss += (loss.item())
                # total_loss += loss_2.item()
                
                # Logging loss per step
                if step % self.log_step == 0:
                    average_loss = total_loss / (step + 1)
                    list_loss.append({
                        'step': step,
                        'loss': loss,
                    })
                    print(f"\nEpoch {epoch + 1}, Step {step + 1}/{len(self.train_dataloader)}, Loss: {average_loss}\n")

                # Save model per step
                if step % self.save_step == 0:
                    print(f'Save model with: {step} steps')
                    model_path = f"{config.version}_epoch_{epoch + 1}_step_{step + 1}.pth"
                    torch.save(self.model.state_dict(), model_path)


            # In thông tin sau mỗi epoch
            average_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss}")
            
        json.dump(list_loss, open('./list_loss.json', 'w'), ensure_ascii=False)
        print("Training finished.")

