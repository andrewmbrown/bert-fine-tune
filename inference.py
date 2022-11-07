from utils import *
from bertTokenizer import *
from bert import *

# initialize tokenizer
tokenizer = initialize_tokenizer()

# initialize model to load weights into
model_config = config_model(train_data_len=0, path_to_config='.', config_file='model_config.yaml')
model, optimizer, scheduler = init_model(**model_config)

# load model from saved weights
model = load_model(model, path_to_weights='./model_save', weights_file='training_args.bin', device='cpu')

# inference and test model



