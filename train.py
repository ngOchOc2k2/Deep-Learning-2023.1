
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)



class MyTrainer():
    def __init__(self, model, tokenizer, train_dataloader, epochs, loss_fct, scheduler, use_amp,
                 optimizer_class, optimizer_params, weight_decay, evaluation_steps, 
                 output_path, save_best_model, max_grad_norm, show_progress_bar, callback, device=None):
        
        self.model = model
        self.tokenizer = model
        
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        
        self.loss_fct = loss_fct
        self.scheduler = scheduler
        
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        
        self.weight_decay = weight_decay
        self.evaluation_steps = evaluation_steps
        
        self.output_path = output_path
        self.save_best_model = save_best_model
        
        self.max_grad_norm = max_grad_norm
        self.show_progress_bar = show_progress_bar
        
        self.callback = callback
        self.use_amp = use_amp
        
        self.activation_fct = nn.GELU()
        
        # train_dataloader.collate_fn = self.smart_batching_collate

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)


        self.best_score = -9999999
        self.num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        
        
    def train(self):
        skip_scheduler = False
        for epoch in trange(self.epochs, desc="Epoch", disable=not self.show_progress_bar):
            training_steps = 0
            total_loss = 0
            self.model.zero_grad()
            self.model.train()

            for features in tqdm(self.train_dataloader, desc="Iteration", smoothing=0.05):
                if self.use_amp:
                    with self.autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        # logits = self.activation_fct(model_predictions.logits)
                        print(model_predictions)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        # loss_value = self.loss_fct(logits, labels)

                    scale_before_step = self.scaler.get_scale()
                    # self.scaler.scale(loss_value).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    skip_scheduler = self.scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = self.activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    # loss_value = self.loss_fct(logits, labels)
                    # loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                # total_loss += loss_value.item()

                if not skip_scheduler:
                    self.scheduler.step()

                training_steps += 1
                
            print(f"Epoch {epoch + 1}/{self.epochs}")

            #     if self.evaluator is not None and self.evaluation_steps > 0 and training_steps % self.evaluation_steps == 0:
            #         self._eval_during_training(self.evaluator, self.output_path, self.save_best_model, epoch, training_steps, self.callback)

            #         self.model.zero_grad()
            #         self.model.train()

            # if self.evaluator is not None:
            #     self._eval_during_training(self.evaluator, self.output_path, self.save_best_model, epoch, -1, self.callback)


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        return self.save(path)