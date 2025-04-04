# Note: you will need to install huggingface_hub, fastparquet
#import fastparquet

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/tattabio/convergent_enzymes/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/tattabio/convergent_enzymes/" + splits["test"])

print(df_train.head())
print(df_test.head())