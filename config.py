from configparser import ConfigParser



EPOCHS = 20
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
ADAPTER_BOTTLENECK = 256


class Configs:
    # data_path = '/home/luungoc/BTL-2023.1/Deep learning/collectionandqueries/Data Json/train_data.json'
    
    data_path = '/kaggle/input/data-macro/Data Json/train_data.json'
    tokenizer = 'bert-base-uncased'
    max_length = 256
    truncation = True
    device = 'cuda'
    adapter = True
    
    model = 'bert-base-uncased'
    dropout = 0.1
    output_size = 768
    freeze_encoders = 'adapter'
    
    lr = 1e-4
    save_step = 400
    use_amp = True
    log_step = 10
    optimizer_params = None
    weight_decay = 0.1
    evaluation_steps = None 
    save_path = None
    save_best_model = None 
    max_grad_norm = None
    show_progress_bar = True 
    epoch = 3
    batch_size = 8
    version = 'version_1'