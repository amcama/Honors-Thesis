import random
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from sklearn.metrics import accuracy_score
from clulab.clu_tokenizer import CluTokenizer
from clulab.names import Names
from IPython.display import display
from datasets import Dataset, DatasetDict

words = [ "Notably", ",", "overexpressing", "MafB", "in", "human", "beta-cell", "lines", "(", "beta", "TC3", "cells", ")", "resulted", "in", "increased", "cell", "proliferation", "by", "upregulating", "important", "cell", "cycle", "regulators", ",", "like", "cyclin", "D2", "and", "cyclin", "B", "(", "28", ")", "." ]

# https://github.com/clulab/scala-transformers/blob/main/encoder/src/main/python/test_clu_tokenizer.py
transformer_name = 'bert-base-cased'
tokenizer = CluTokenizer.get_pretrained(transformer_name)

output = tokenizer(words, is_split_into_words=True) 

input_ids = output.input_ids

tokens = tokenizer.convert_ids_to_tokens(input_ids)

print()
print("~ TOKENS:", tokens)
print()
print("~ INPUT_IDS:", input_ids)

# display results
train_df = pd.DataFrame(
    [tokens, input_ids],
    index=['tokens', 'input_ids']
)
print("\n~ Displaying Pandas DataFrame...")
display(train_df)


# 13.2 
ds = DatasetDict()
ds['train'] = Dataset.from_pandas(train_df)