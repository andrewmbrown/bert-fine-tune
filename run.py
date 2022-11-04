import yaml
import sys
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
# read application_config.yaml file and populate config_dict
with open("./model_config.yaml", "r") as stream:
    try:
        config_dict = yaml.safe_load(stream)

    except yaml.YAMLError as e:
        print(e)
        sys.exit(1)

try:
    model_config = {
        'model_name' : config_dict['model_config']['model_name'],
        'num_labels' : config_dict['model_config']['num_labels'],
        'output_attentions' :config_dict['model_config']['output_attentions'], 
        'output_hidden_states' :config_dict['model_config']['output_hidden_states'],
        'use_cuda' : config_dict['model_config']['use_cuda'],
        'lr' : config_dict['model_config']['lr'],
        'eps' : config_dict['model_config']['eps'],
        'train_length' : len(train_dataloader),
        'epochs' : config_dict['model_config']['epochs'],
        'num_warmup_steps' : config_dict['model_config']['num_warmup_steps'],
    }

except KeyError as e:
    print("Error reading model config file entries.")
    print(e)
    sys.exit(1)

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

