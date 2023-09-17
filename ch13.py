import random
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split


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
    # read json file
    df = pd.read_json(filename)
    return df

train_df = read_data('sample_training_data/PMC544947-classifictaion-out.json')

print(train_df)

train_df, eval_df = train_test_split(train_df, train_size=0.2)
# train_df.reset_index(inplace=True, drop=True)
# eval_df.reset_index(inplace=True, drop=True)
# print(f'train rows: {len(train_df.index):,}')
# print(f'eval rows: {len(eval_df.index):,}')
