import pandas as pd
import csv
import json
from tqdm.auto import tqdm
tqdm.pandas()

# Read data
df = pd.read_csv('cpv/all.csv', sep=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, usecols=[1, 2, 3], dtype={'cpv': object})
df.dropna(inplace=True)
df['cpv'] = df['cpv'].progress_apply(lambda v: '0' * (8 - len(v)) + v)

cpv_hierarchy = json.load(open('cpv/cpv_hierarchy.json', 'r'))

def get_all_cpv_codes(node):
    codes = set([node['cpv']])
    for child in node['children']:
        codes = codes.union(get_all_cpv_codes(child))
    return codes
cpv = get_all_cpv_codes({'cpv': 'Root', 'children': cpv_hierarchy })

# Add labels and tokens
def int_to_hierarchy(v):
    labels = [v[:2] + '0' * (len(v) - 2)]
    for i in range(2, len(v)):
        if v[i] == '0':
            break
        labels.append(v[:i + 1] + '0' * (len(v) - i - 1))
    return [l for l in labels if l in cpv]

df['label'] = df.cpv.progress_apply(int_to_hierarchy)

df.desc = df.desc.str.replace(r'\d{8}(-\d)?', '', regex=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EuropeanParliament/EUBERT")
df['token'] = df.desc.progress_apply(lambda x: tokenizer.tokenize(x))

# Split data into train, val, test
import numpy as np

train, validate, test = np.split(df[['label', 'token']].sample(frac=1, random_state=42), [int(.7 * len(df)), int(.85 * len(df))])
train.to_json('cpv/cpv_train.json', orient='records', lines=True)
validate.to_json('cpv/cpv_val.json', orient='records', lines=True)
test.to_json('cpv/cpv_test.json', orient='records', lines=True)
