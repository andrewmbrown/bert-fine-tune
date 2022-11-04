import json
from bertTokenizer import *
from utils import *
from bert import *
from train import *

# 1. Run GPU test and get device
device = print_cuda_diagnostic()

# 2. Pre-process and Tokenize data
path_to_data = './data/data.json'
# preprocess data
df, i_to_label = pre_proc(path_to_data)
# initialize tokenizer
bert_tokenizer = initialize_tokenizer()
# tokenize data
dataset = tokenize_dataset(bert_tokenizer, df)
# split into train and test
val_size = 0.1
train_dataloader, validation_dataloader = train_test_split(dataset, val_size)

# 3. Instantiate Model, optimizer, scheduler
model_config = {
    'model_name' : 'bert-base-uncased',
    'num_labels' : 8,
    'output_attentions' : True,
    'output_hidden_states' : True,
    'use_cuda' : False,
    'lr' : 2e-5,
    'eps' : 1e-8,
    'train_length' : len(train_dataloader),
    'epochs' : 3,
    'num_warmup_steps' : 0
}
model, optimizer, scheduler = init_model(**model_config)

# 3.5 Print Model Diagnostics
get_model_info(model)
get_state_dict(model, optimizer)

# 4. Train Model
training_stats = train_model(model, optimizer, scheduler, train_dataloader, validation_dataloader, model_config['epochs'], device)

# 5. Analyze training stats
analyze_train_stats(training_stats, show=False, save=True, save_path='./figs')

# 6. Validate Model

# 7. Save model weights to load later
save_model(model, bert_tokenizer)

