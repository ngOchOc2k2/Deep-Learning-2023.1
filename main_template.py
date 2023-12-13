import argparse
import random
from sampler import data_sampler
from config import Config
import torch
from model.bert_encoder import Bert_Encoder
from model.dropout_layer import Dropout_Layer
from model.classifier import Softmax_Layer, Proto_Softmax_Layer
from data_loader import get_data_loader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import collections
from copy import deepcopy
import os
import logging

# Configure logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="tacred", type=str)
    parser.add_argument("--shot", default=10, type=str)
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    config.task = args.task
    config.shot = args.shot
    config.step1_epochs = 5
    config.step2_epochs = 15
    config.step3_epochs = 20
    config.temperature = 0.08

    if config.task == "FewRel":
        config.relation_file = "data/fewrel/relation_name.txt"
        config.rel_index = "data/fewrel/rel_index.npy"
        config.rel_feature = "data/fewrel/rel_feature.npy"
        config.rel_des_file = "data/fewrel/relation_description.txt"
        config.num_of_relation = 80
        if config.shot == 5:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_5/test_0.txt"
        elif config.shot == 10:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_10/test_0.txt"
        else:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_2/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_2/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_2/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_2/test_0.txt"
    else:
        config.relation_file = "data/tacred/relation_name.txt"
        config.rel_index = "data/tacred/rel_index.npy"
        config.rel_feature = "data/tacred/rel_feature.npy"
        config.num_of_relation = 41
        if config.shot == 5:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_5/test_0.txt"
        else:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_10/test_0.txt"

    result_cur_test = []
    result_whole_test = []
    bwt_whole = []
    fwt_whole = []
    X = []
    Y = []    
    relation_divides = []
    for i in range(10):
        relation_divides.append([])
    for rou in range(config.total_round): # run multiple round with different seed and take average
        test_cur = []
        test_total = []
        
        random.seed(config.seed + rou * 100)
        
        # Load data
        sampler = data_sampler(config=config, seed=config.seed+rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        id2sentence = sampler.get_id2sent()
        num_class = len(sampler.id2rel)
        
        # Load model
        """
        Not implemented
        """
        
        # Config memory
        # them memory gi thi them vao dictionary nay
        memory = {
            'history_relations' : [],
            'prev_history_relations' : [],
            
            'history_prototypes' : [],
            'prev_history_prototypes': [],
            
            'history_typical_samples' : [],
            'prev_history_typical_samples': []
            
        }
        """
        Not implemented
        """
        
        # Loop through tasks
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            logging.debug("Current relation : ")
            logging.debug(current_relations)
            
            # update memory
            memory['prev_history_relations'] = memory['history_relations']
            
            for relation in current_relations : 
                memory['history_relations'].append(relation)
            
            """
            Addition implementation here
            """
            
            #Train
            """
            Addition implememtation here
            """
            
            # Evaluate after training on this task
            cur_acc = 0  # eval on this task (on test set)
            total_acc = 0 # eval on all seen tasks (on test set)
            
            logging.debug(f'Restart Num : {rou + 1}')
            logging.debug(f'task--{steps + 1}:')
            logging.debug(f'current test acc:{cur_acc}')
            logging.debug(f'history test acc:{total_acc}')
            
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            
        # Averate result of all rounds
        result_cur_test.append(np.array(test_cur))
        result_whole_test.append(np.array(test_total)*100)
        logging.debug("result_whole_test")
        logging.debug(result_whole_test)
        avg_result_cur_test = np.average(result_cur_test, 0)
        avg_result_all_test = np.average(result_whole_test, 0)
        logging.debug("avg_result_cur_test")
        logging.debug(avg_result_cur_test)
        logging.debug("avg_result_all_test")
        logging.debug(avg_result_all_test)
        std_result_all_test = np.std(result_whole_test, 0)
        logging.debug("std_result_all_test")
        logging.debug(std_result_all_test)