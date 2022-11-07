import torch
import json
import pandas as pd
import transformers
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

def pre_proc(path_to_data):
    """
    Load data from json file and convert to pandas dataframe
    encode categorical data to integer labels
    """
    with open(path_to_data, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data=data['utterances'])
    # df has 4 columns: 'text', 'entites', 'traits', 'intent'
    # 'text' is a list of strings
    # 'entities' is a list of dicts - NOT USED
    df.drop(columns=['entities'], inplace=True)
    # 'traits' is a list of dicts - NOT USED
    df.drop(columns=['traits'], inplace=True)
    # 'intent' is 1 of 8 strings
    # we want our intents to be encoded as 0-7
    # cast column to category, then get the category codes
    df["intent"] = df["intent"].astype('category')
    df["intent_label"] = df["intent"].cat.codes
    # also return dict of intent to label, for later decoding
    i_to_label = dict(enumerate(df['intent'].cat.categories))

    # Get one hot encoding of intent column
    df_one_hot_intents = pd.get_dummies(df['intent'])
    # loop through intents to squeeze into one entry each
    one_hot_intents = []
    for idx, row in df_one_hot_intents.iterrows():
        one_hot_intents.append(row.values)
    
    # Join the encoded df
    df['one_hot_labels'] = one_hot_intents

    return df, i_to_label
    
def initialize_tokenizer():
    # [SEP] - seperator token between sentences
    # [CLS] - classification token On the output of the final (12th) transformer
    # only the first embedding (corresponding to the [CLS] token) is used by the classifier.
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer

def tokenize_dataset(tokenizer, df):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for idx, row in df.iterrows():
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            row['text'],                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 100,           # Pad & truncate all sentences.
                            truncation=True,             # Truncate sentences longer than 100 tokens
                            padding = 'max_length',    # Pad sentences to max_length
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = df['one_hot_labels'].tolist()
    # labels are dtype float 32 for consistency (enncoder outputs float 32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', df['text'][0])
    print('Token IDs:', input_ids[0])

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset
