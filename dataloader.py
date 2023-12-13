import torch
import json
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

class MyDataset(Dataset):
    def __init__(self, config):
        self.data = self.load_data(config.data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self.max_length = config.max_length
        self.truncation = config.truncation

        if config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(config.device))

        
    def load_data(self, data_path):
        data = json.load(open(data_path, 'r'))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = {key: value for key, value in self.data[idx].items() if key != 'label'}
        neg = []
        for item in text['negative_text']:
            neg.append(item['P_text']['Description'])
            
        return {
            'query_text': text['query_text']['Q_text'],
            'positive_text': text['positive_text']['P_text']['Description'],
            'negative_text': neg,
        }
