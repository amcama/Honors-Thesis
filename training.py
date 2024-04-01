import os, json
import pandas as pd
import random
import torch
import numpy as np
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import AutoConfig
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import copy
import os
from transformers import DataCollatorForTokenClassification


transformer_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)

configuration = 3
ignore_index = -100
classes = ['Negative_activation', 'Positive_activation', 'No_relation']

TOKENIZERS_PARALLELISM=False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    init()
    config = AutoConfig.from_pretrained(transformer_name, num_labels = len(classes), output_hidden_states=True)
    global model
    model = (BertForSequenceClassification.from_pretrained(transformer_name, config=config))
    tokenizer.add_special_tokens({'additional_special_tokens': ['[E1]','[/E1]', '[E2]', '[/E2]']})
    model.resize_token_embeddings(len(tokenizer))

    data = read_data()
    # maxpool(data['regulations']) # should i be doing this as well for data without entity markers/multiple entity markers? 


    x = data.values()
    data_list = []
    for e in x:
       for f in e:
           data_list.append(f)

    data = data_list

    # Take the first 60% as train, next 20% as development, last 20% as test 
    train_df, eval_df = train_test_split(data, train_size=0.6)
    eval_df, test_df = train_test_split(eval_df, train_size=0.5)

    print(f'train rows: {len(train_df):,}')
    print(f'eval rows: {len(eval_df):,}')
    print(f'test rows: {len(test_df):,}')

    # create train dataset
    train_df = pd.DataFrame(train_df)
    eval_df = pd.DataFrame(eval_df)
    test_df = pd.DataFrame(test_df)

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(eval_df)
    ds['test'] = Dataset.from_pandas(test_df)

    

    train_ds = ds['train'].map(
        tokenize, batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )   

    
    # TODO add condition for configuration 

    eval_ds = ds['validation'].map(
        tokenize,
        batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )
    eval_ds.to_pandas()

    
    num_epochs = 2
    batch_size = 24 # change this for smaller data sets
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
    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )    

    print("\n", train_ds, "\n")
    # exit(0)
    train_output = trainer.train()
    exit(0)
    print("> Train Output:\n", train_output, "\n")

    test_ds = ds['test'].map(
        tokenize,
        batched=True,
        remove_columns=['event_indices', 'polarity', 'controller_indices', 'controlled_indices', 'trigger_indices']
    )
    test_ds.to_pandas()

    output = trainer.predict(test_ds)

    print("test output: ", output)

    y_true = output.label_ids
    y_pred = np.argmax(output.predictions, axis=-1)
    target_names = classes
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("y_true: ", y_true)
    print("y_pred: ", y_pred)

    print("\nConfusion Matrix:")
    cm = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1,2])
    print(cm)

    
def tokenize(examples):
    output = tokenizer(examples['sentence_tokens'], is_split_into_words=True, truncation=True)
    return output
            

def add_entity_markers(data):
    count = 0
    for e in data['regulations']:
        og_sentence = e['sentence_tokens']
        new_sentence = copy.deepcopy(e['sentence_tokens'])

        controller_indices = e['controller_indices']
        controlled_indices = e['controlled_indices']

        controller_start = controller_indices[0]
        controller_end = controller_indices[-1]

        controlled_start = controlled_indices[0]
        controlled_end = controlled_indices[-1]

        if (controller_start < controlled_start):
            new_sentence.insert(controller_start, "[E1]")
            new_sentence.insert(controller_end + 1, "[/E1]" )
            new_sentence.insert(controlled_start+2, "[E2]")
            new_sentence.insert(controlled_end+3, "[/E2]")

        else:
            new_sentence.insert(controlled_start, "[E1]")
            new_sentence.insert(controlled_end + 1, "[/E1]" )
            new_sentence.insert(controller_start+2, "[E2]")
            new_sentence.insert(controller_end+3, "[/E2]")
        e['sentence_tokens'] = new_sentence
        # print("\n--------")
        # print("Regulations")
        # print(controller_indices)
        # print(controlled_indices)
        # print("og sentence: ", og_sentence, "\n")
        # print("new sentence: ", new_sentence, "\n")
        # print()


    for e in data['hardInstances']:
        og_sentence = e['sentence_tokens']
        new_sentence = copy.deepcopy(e['sentence_tokens'])
        indices = e['entities_indices']
        num_entities = len(indices)
        offset = 0
        for i in range(0, num_entities):
            if (offset == 0):
                start = indices[i][0] 
                end = indices[i][-1]
                offset += 1 
            else: 
                start = indices[i][0] + offset + 1
                end = indices[i][0] + offset + 2
                offset += 2

            new_sentence.insert(start, "[E{}]".format(i+1))
            new_sentence.insert(end+1, "[/E{}]".format(i+1))
        e['sentence_tokens'] = new_sentence

        # print("\n--------")
        # print("Hard Instances")
        # print(indices, "\n")
        # print("og sentence: ", og_sentence, "\n")
        # print("new sentence: ", new_sentence, "\n")
        # print()

    for e in data['withoutRegulations']:
        og_sentence = e['sentence_tokens']
        new_sentence = copy.deepcopy(e['sentence_tokens'])
        indices = e['entities_indices']
        num_entities = len(indices)
        offset = 0
        for i in range(0, num_entities):
            if (offset == 0):
                start = indices[i][0] 
                end = indices[i][-1]
                offset += 1 
            else: 
                start = indices[i][0] + offset + 1
                end = indices[i][0] + offset + 2
                offset += 2

            new_sentence.insert(start, "[E{}]".format(i+1))
            new_sentence.insert(end+1, "[/E{}]".format(i+1))
        e['sentence_tokens'] = new_sentence

        # print("\n--------")
        # print("Without Regulations")
        # print(indices, "\n")
        # print("og sentence: ", og_sentence, "\n")
        # print("new sentence: ", new_sentence, "\n")
        # print()


    return data



def read_test_data():
    json_data = []
    f = open("test_data.json")
    data = json.load(f)
    # print(data)

    for e in data:
        if (e.get('type')):
            e['label'] = e.pop('type')
            if (e['label'] == "Negative_activation"):
                e['label'] = 0
            elif (e['label'] == "Positive_activation"):
                e['label'] = 1
            elif (e['label'] == "Negative_regulation"):
                e['label'] = 0
            elif (e['label'] == "Positive_regulation"):
                e['label'] = 1
        else:
            e['label'] = 2
    if (configuration != 1):
        add_entity_markers(data)
    return data
 

def read_data():
    """ 
    Read data from sample_training_data folder and remove duplicates.
    """
    data = {
        "regulations" : [],
        "hardInstances": [],
        "withoutRegulations": [],
        "emptySentences": []
        }
    
    # directory1 = 'sample_training_data'
    # directory2 = 'negative_training_data'

    directory1 = 'test_sample'
    directory2 = 'test_negative'

    negative_count = 0
    positive_count = 0
    none_count = 0

    count = 0

    for filename in os.listdir(directory1):
        if filename.endswith('.json'):
            with open(os.path.join(directory1, filename)) as f:
                file = json.load(f)
                if (len(file) > 0):
                    for e in file:
                        if (e['type'] == "Negative_activation" or e['type'] == "Negative_regulation"):
                                e['label'] = 0
                                negative_count += 1
                        elif (e['type'] == "Positive_activation" or e['type'] == "Positive_regulation"):
                                e['label'] = 1
                                positive_count += 1
                
                        data['regulations'].append(e)

    for filename in os.listdir(directory2):
        if filename.endswith('.json'):
            with open(os.path.join(directory2, filename)) as f:
                file = json.load(f)
                if (len(file) > 0): 
                    regulations = file['regulations']
                    hard_instances = file['hardInstances']
                    without_regulations = file['withoutRegulations']
                    empty_sentences = file['emptySentences']

                    for e in regulations:
                        if (e['type']):
                            if (e['type'] == "Negative_activation" or e['type'] == "Negative_regulation"):
                                e['label'] = 0
                                negative_count += 1
                            
                            elif (e['type'] == "Positive_activation" or e['type'] == "Positive_regulation"):
                                e['label'] = 1
                                positive_count += 1
                            
                            else:
                                raise Exception("Unknown label in regulations data!")
                            data['regulations'].append(e)
                            count += 1
                        else:
                            raise Exception("Unknown label in regulations data!")

                    for e in hard_instances:
                        e['label'] = 0
                        data['hardInstances'].append(e)
                        count += 1
                        negative_count += 1 

                    for e in without_regulations:
                        e['label'] = 2
                        none_count += 1
                        data['withoutRegulations'].append(e)
                        count += 1

                    for e in empty_sentences:
                        e['label'] = 2
                        none_count += 1
                        data['emptySentences'].append(e)
                        count += 1 
                    

    # print("Positive: ", positive_count)
    # print("Negative: ", negative_count)
    # print("None: ", none_count)
    
    # TODO remove duplicates & shuffle 
    
    if (configuration != 1):
        data = add_entity_markers(data)
    else:
        data = data
    return data


def pretty_print(list):
    for e in list:
        print(e, "\n")

def maxpool(e1, e2, e3, e4):
    # replace concat with a function that does max pooling of the four
    # combine embeddings in 3 dif ways : could do max pooling and have an embedding thats the size of each individual one
    # could also do an average
    # reduce the size of the linear layer input because now its 4 times the size of hidden state and needs to be og size
    # print("individual embeddings shape: ", e1.shape)
    
    # combine embeddings
    concatenated = torch.stack((e1,e2,e3,e4))
    maxpooled = torch.max_pool1d(concatenated, kernel_size=1) # what should kernel size be?
    print(maxpooled.shape)
    return maxpooled
    
     


# https://github.com/huggingface/transformers/blob/65659a29cf5a079842e61a63d57fa24474288998/src/transformers/models/bert/modeling_bert.py#L1486
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
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
        embeddings = outputs.last_hidden_state
        
        final_vector = torch.Tensor()
        
        for i in range(0, len(input_ids)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            # print()
            # print(tokens)
            entity_indexes = dict()
            for j in range(0, len(tokens)):
                if (tokens[j] == "[E1]"):
                    entity_indexes["[E1]"] = j
                if (tokens[j] == "[/E1]"):
                    entity_indexes["[/E1]"] = j
                if (tokens[j] == "[E2]"):
                    entity_indexes["[E2]"] = j
                if (tokens[j] == "[/E2]"):
                    entity_indexes["[/E2]"] = j
            if len(entity_indexes) < 4:
                pass # TODO 
            else:
                e1 = embeddings[i][entity_indexes.get("[E1]")]
                e2 = embeddings[i][entity_indexes.get("[/E1]")]
                e3 = embeddings[i][entity_indexes.get("[E2]")]
                e4 = embeddings[i][entity_indexes.get("[/E2]")] 
                returned = maxpool(e1, e2, e3, e4)
                final_vector = torch.concat((final_vector, returned))
                # print(final_vector.shape)                

        # replace concat with a function that does max pooling of the four
        # combine embeddings in 3 dif ways : could do max pooling and have an embedding thats the size of each indiidual one
        # could also do an average
        # reduce the size of the linear layer input because now its 4 times the size of hidden state and needs to be og size
        
        sequence_output = self.dropout(final_vector) 
        logits = self.classifier(sequence_output) # linear layer? 

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()

            print("logits shape: ", logits.shape)
            print("labels shape: ", labels.shape)

            inputs = logits.view(-1)
            targets = labels.view(-1)

            print("Inputs Shape: ", inputs.shape)
            print("Targets Shape: ", targets.shape)
            print()

            loss = loss_fn(inputs, targets)            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


def compute_metrics(eval_pred):
    print("COMPUTING METRICS")
    y_true = eval_pred.label_ids
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    return {'accuracy': accuracy_score(y_true, y_pred)}

def init():
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
main()