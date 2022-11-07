import datetime
import numpy as np
import torch
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler, random_split, RandomSampler

def print_cuda_diagnostic():
    # If there's a GPU available...
    if torch.cuda.is_available():    
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def get_model_info(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print("\n\n")

def get_state_dict(model, optimizer):
    """
    Similar to get model info, but for torch state dictionary
    printing state dictionary for model and optimzier
    """
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("\n\n")
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    print("\n\n")

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    """
    Predictions are an array of len 8, each num corresponding to 1 intent
    Labels are one-hot encoded, each idx corresponding to the label
    """
    # from predictions, take largest prob, this is our prediction
    pred_flat = np.argmax(preds, axis=1).flatten()
    # from one-hot labels, return the index (this is our answer)
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_test_split(dataset, val_size=0.1):
    # Create a N:M train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int((1-val_size) * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
    # size of 16 or 32.
    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader, validation_dataloader

def analyze_train_stats(training_stats, show=False, save=False, save_path='./figs'):
    # Create output directory if needed
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # display all floats within two decimal places
    pd.set_option('precision', 2)
    # create dataframe of all training stats
    df = pd.DataFrame(data=training_stats)
    # epoch is row index here
    df = df.set_index('epoch')

    # Plot the learning curve.
    plt.plot(df['Training Loss'], 'b-o', label="Training")
    plt.plot(df['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    title = "Training & Validation Loss"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    if show: plt.show()
    if save: plt.savefig(f"{save_path}/{title}.png")

def config_model(train_data_len, path_to_config='.', config_file='model_config.yaml'):
    """
    configure model dictionary from yaml file
    only setting up dict, not instantiating model
    """
    # read application_config.yaml file and populate config_dict
    with open(f"{path_to_config}/{config_file}", "r") as stream:
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
            'train_length' : train_data_len,
            'epochs' : config_dict['model_config']['epochs'],
            'num_warmup_steps' : config_dict['model_config']['num_warmup_steps'],
        }

    except KeyError as e:
        print("Error reading model config file entries.")
        print(e)
        sys.exit(1)
    
    return model_config

def save_model(model, tokenizer, output_dir='./model_save/'):
    # we can reload the model using from_pretrained()

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # also save using pytorch method
    torch.save(model.state_dict(), os.path.join(output_dir, 'training_args.bin'))

def load_model(model, path_to_weights='.', weights_file='training_args.bin', device='cpu'):
    """
    using saved state_dict and loaded empty (pretrained) model, we load weights
    first we load state_dict, then we load that state dict into our model
    """
    try:
        state_dict = torch.load(f"{path_to_weights}/{weights_file}", map_location=torch.device(device))
    except Exception as e:
        print("Model not loaded successfully")
        print(e)
    
    # state_dict contains our trained and finetuned model weights - load them onto a base model
    model.load_state_dict(state_dict)   
    return model
    
def inference(model, tokenizer, input=None, device='cpu'):
    # set model to eval model, e.g. dropout and batch normalization are turned off
    model.eval()
    # allow for either pre-written inputs, or live typing
    if input is not None:
        for i in input:
            encoded_dict = tokenizer.encode_plus(i,
                                                add_special_tokens=True,
                                                max_length=100,
                                                padding='max_length',
                                                return_attention_mask=True, 
                                                return_tensors='pt'
                                                )
            # grab input id tokens for model
            tokens = encoded_dict['input_ids']

            # grab attention mask
            attention_mask = encoded_dict['attention_mask']

            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = model(tokens,
                                token_type_ids=None,
                                attention_mask=attention_mask)
            
            # obtain prediction logits and mount them on cpu device
            logits = outputs['logits']
            logits = logits.detach().cpu.numpy()

            pred = np.argmax(logits, dim=0)

    # live typing inputs to model from stdin
    else:
        while True:
            print("Enter text for classifier (enter STOP to end)")
            input = sys.stdin
            if input == "STOP":
                break
            encoded_dict = tokenizer.encode_plus(i,
                                    add_special_tokens=True,
                                    max_length=100,
                                    padding='max_length',
                                    return_attention_mask=True, 
                                    returnn_tensors='pt'
                                    )
            # grab input id tokens for model
            tokens = encoded_dict['input_ids']

            # grab attention mask
            attention_mask = encoded_dict['attention_mask']

            # convert tokens and masks to tensors
            tokens = torch.cat(tokens, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)

            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad:
                outputs = model(tokens,
                                token_type_ids=None,
                                attention_mask=attention_mask)
            
            # obtain prediction logits and mount them on cpu device
            logits = outputs['logits']
            logits = logits.detach().cpu.numpy()

            pred = np.argmax(logits, dim=0)
    return
    
