from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from utils import *
import random

def init_model(model_name: str,
                num_labels: int,
                output_attentions: bool,
                output_hidden_states: bool,
                use_cuda: bool,
                lr: float,
                eps: float,
                train_length: int,
                epochs: int,
                num_warmup_steps: int):
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        model_name, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = num_labels, # The number of output labels
        output_attentions = output_attentions, # Whether the model returns attentions weights.
        output_hidden_states = output_hidden_states, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    # this line is important for the optimizer, its initialized differently
    # based on which device the model situates
    if use_cuda: model.cuda()
    else: device = torch.device("cpu")

    # initialize our optimizer
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = eps # args.adam_epsilon  - default is 1e-8.
                    )

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    # epochs between 2-4
    total_steps = train_length * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = num_warmup_steps, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    return model, optimizer, scheduler