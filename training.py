import random
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from sklearn.metrics import accuracy_score
from clulab.clu_tokenizer import CluTokenizer
from clulab.names import Names

# enable tqdm in pandas
tqdm.pandas()

# set to True to use the gpu (if there is one available)
use_gpu = True

# select device
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
print(f'device: {device.type}')

# random seed
seed = 1234

# set random seed
if seed is not None:
    print(f'random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- [2] ---
def read_data(filename):
    df = pd.read_json(filename)
    return df

# --- [3] ---
labels = ['sentence_tokens', 'event_indices', 'type', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
# train_df = read_data('sample_training_data/PMC544947-classifictaion-out.json')
train_df = read_data('example.json')
test_df = read_data('test.json')

# --- [4] ---
train_df, eval_df = train_test_split(train_df, train_size=0.9)
train_df.reset_index(inplace=True, drop=True)
eval_df.reset_index(inplace=True, drop=True)

print(f'train rows: {len(train_df.index):,}')
print(f'eval rows: {len(eval_df.index):,}')
print(f'test rows: {len(test_df.index):,}')
print()

# --- [5] ---
ds = DatasetDict()
ds = DatasetDict()
ds['train'] = Dataset.from_pandas(train_df)
ds['validation'] = Dataset.from_pandas(eval_df)
ds['test'] = Dataset.from_pandas(test_df)
# print(ds)
# print()

# --- [6] ---

# --- [7] ---
# https://github.com/clulab/scala-transformers/blob/main/encoder/src/main/python/test_clu_tokenizer.py

transformer_name = "bert-base-cased"
tokenizer = CluTokenizer.get_pretrained(transformer_name)


def tokenize(examples):
    sentences = (examples['sentence_tokens'])    
    tokenized_words = tokenizer(sentences, is_split_into_words=True)
    
    ids_from_words = tokenized_words.input_ids
    tokens_from_words = tokenizer.convert_ids_to_tokens(ids_from_words)
    
    print(tokens_from_words)
    print(ids_from_words)

# TODO need to figure out how to get batches working
train_ds = ds['train'].map(tokenize, batched=False)  # changed batched=True to batched=False
eval_ds = ds['validation'].map(tokenize, batched=False)  # changed batched=True to batched=False
train_ds.to_pandas()

# --- [8] ---
# https://github.com/huggingface/transformers/blob/65659a29cf5a079842e61a63d57fa24474288998/src/transformers/models/bert/modeling_bert.py#L1486

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        cls_outputs = outputs.last_hidden_state[:, 0, :]
        cls_outputs = self.dropout(cls_outputs)
        logits = self.classifier(cls_outputs)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# --- [9] --- 
config = AutoConfig.from_pretrained(
    transformer_name,
    num_labels=len(labels),
)

print()
model = (
    BertForSequenceClassification
    .from_pretrained(transformer_name, config=config)
)

# --- [10] ---
num_epochs = 2
batch_size = 24
weight_decay = 0.01
model_name = f'{transformer_name}-sequence-classification'

training_args = TrainingArguments(
    output_dir=model_name,
    log_level='error',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    weight_decay=weight_decay,
)

# --- [11] ---
def compute_metrics(eval_pred):
    y_true = eval_pred.label_ids
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    return {'accuracy': accuracy_score(y_true, y_pred)}


# --- [12] ---
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

# --- [13] ---
trainer.train()
