import os, json
import pandas as pd
import random
import math
import torch
import numpy as np
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict
from clulab.clu_tokenizer import CluTokenizer
from transformers import AutoTokenizer

transformer_name = 'bert-base-cased'
tokenizer = CluTokenizer.get_pretrained(transformer_name)

def main():
    tqdm.pandas()
    use_gpu = True
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    # print(f'device: {device.type}')
    seed = 1234
    if seed is not None:
        # print(f'random seed: {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    data = read_data()
    # pretty_print(data)

    # Take the first 60% as train, next 20% as development, last 20% as test (Don't use test for now)
    size = len(data)
    train_size = round(size * 0.6)
    dev_size = round(size * 0.2)
    test_size = round(size * 0.2)

    # print("size: {}  train_size: {}  dev_size: {}  test_size: {}\n".format(size, train_size, dev_size, test_size))
    assert(train_size + dev_size + test_size == size)

    train_list = []
    dev_list = []
    test_list = []

    for i in range(0, train_size):
        train_list.append(data[i])
        
    for i in range(train_size, (train_size + dev_size)):
        dev_list.append(data[i])

    for i in range(dev_size, (dev_size + test_size)):
        test_list.append(data[i])

    # create train dataset
    train_df = pd.DataFrame(train_list)
    dev_df = pd.DataFrame(dev_list)
    test_df = pd.DataFrame(test_list)

    print(f'train rows: {len(train_df.index):,}')
    print(f'eval rows: {len(dev_df.index):,}')
    print(f'test rows: {len(test_df.index):,}')

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(dev_df)
    ds['test'] = Dataset.from_pandas(test_df)

    train_ds = ds['train'].map(
        tokenize, batched=True,
        remove_columns=['sentence_tokens', 'event_indices', 'type', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )




def tokenize(examples):
    # todo here
    return tokenizer(examples['sentence_tokens'])







def read_data():
    """ 
    Read data from sample_training_data folder and remove duplicates.
    """
    json_data = []
    directory = 'sample_training_data'
    count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)
                if (len(data) > 0):
                    json_data.append(data)
                    count += 1

        if (count == 2): # just for testing smaller data
            break

    list_no_dups = remove_duplicates(json_data)
    random.shuffle(list_no_dups)
    return list_no_dups

def remove_duplicates(list):
    # concatenate all nested items into single list
    single_list = []
    for e1 in list:
        for e2 in e1:
            single_list.append(e2)

    list_no_duplicates = []
    seen = set()

    for e in single_list:
        joined_string = ''.join(e['sentence_tokens'])
        if (joined_string not in seen):
            list_no_duplicates.append(e)

        seen.add(joined_string)

    # print("-   Seen Count: {}".format(len(single_list)))
    # print("- Original Count: {}".format(list))
    # print("-   Unique Count: {}\n".format(len(list_no_duplicates)))

    return list_no_duplicates

def pretty_print(list):
    for e in list:
        print(e, "\n")

main()