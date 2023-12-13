import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer   
from config import *
from .bert_model import BertModel

model_path = [
    'roberta-base',
    'bert-base-uncased',
]


logger = logging.getLogger(__name__)

class DoubleBert(nn.Module):
    
    def __init__(self, config):
        super(DoubleBert, self).__init__()
        if config.adapter == False:
            self.model = AutoModel.from_pretrained(config.model)
        else:
            self.model = BertModel.from_pretrained(config.model)
            
        # self.model_2 = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        

        self.output_size = config.output_size
        self.linear_layer = nn.Linear(self.model.config.hidden_size * 2, config.output_size)
        
        self.dropout = nn.Dropout(p=config.dropout)
        self.activation = nn.GELU()
        
        self.normalize = nn.LayerNorm(config.output_size)
        self.last_output = nn.Linear(config.output_size, 2)
        self.similar = nn.Linear(config.output_size, 1)
        
        self.max_length = config.max_length
        self.softmax = nn.Softmax(dim=1)


        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(config.device))
        self.device = torch.device(config.device)


    def batching_tokenizer(self, batch):
        main_texts = []
        labels = []

        for example in batch:
            main_texts.append(example.main_text.strip())
            labels.append(example.label)
            

        tokenized = self.tokenizer(main_texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self.device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized, labels
    
    
    def batching_tokenizer_text(self, batch):
        main_texts = []

        for example in batch:
            main_texts.append(example.main_text.strip())

        tokenized = self.tokenizer(main_texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized

    
    
    def forward(self, x_1 , x_2):
        token_1 = self.tokenizer(x_1, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt').to(self.device)
        token_2 = self.tokenizer(x_2, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt').to(self.device)
        
        output_1 = self.model(token_1['input_ids'].to(self.device), attention_mask=token_1['attention_mask'].to(self.device)).last_hidden_state[:, 0, :]
        output_2 = self.model(token_2['input_ids'].to(self.device), attention_mask=token_2['attention_mask'].to(self.device)).last_hidden_state[:, 0, :]
        
        concatenated_output = torch.cat((output_1, output_2), dim=1)
 
        similarity_score = self.linear_layer(concatenated_output)
        similarity_score = self.normalize(similarity_score)
        similar = self.similar(similarity_score)
        
        similarity_score = self.last_output(similarity_score)
        soft_max = self.softmax(similarity_score)
        
        return similar, soft_max
    
    def load_state_model(self, state_path, device='cpu'):
        state_dict = torch.load(state_path, map_location=torch.device(device))

        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        
        return self.model
    
    
# if __name__ == '__main__':
#     model = DoubleBert(model_path='bert-base-uncased', tokenizer_path='bert-base-uncased', output_size=1024)
#     dataset = MyDataset(data_path='/home/luungoc/Code Law/P3_Lawfix_QA_Neg.json',
#                             tokenizer='bert-base-uncased',
#                             max_length=256)

    # while True:
    #     input_text = input('Continue:')
        
    #     if input_text == 'N':
    #         break
    #     else:
    #         text_1 = input('Text 1:')
    #         text_2 = input('Text 2:')
            
    #         print(model(x_1=text_1, x_2=text_2))
        
        