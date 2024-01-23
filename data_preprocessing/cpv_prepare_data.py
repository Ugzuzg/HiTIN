import pandas as pd
import csv
import json
from tqdm.auto import tqdm
tqdm.pandas()

# Read data. 1 - description, 2 - language, 3 - cpv code. 
# Read 'cpv' as object, otherwise pandas converts it to integers removing leading zeros
df = pd.read_csv('cpv/all.csv', sep=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, usecols=[1, 2, 3], dtype={'cpv': object})

# Some descriptions are "None". We drop them, considering that we have sufficient amount of data
df.dropna(inplace=True)

cpv_hierarchy = json.load(open('cpv/cpv_hierarchy.json', 'r'))

def get_all_cpv_codes(node):
    codes = set([node['cpv']])
    for child in node['children']:
        codes = codes.union(get_all_cpv_codes(child))
    return codes
cpv = get_all_cpv_codes({'cpv': 'Root', 'children': cpv_hierarchy })

# Add labels and tokens
# TODO: check if 'Root' needs to be included as a value
def cpv_to_label_hierarchy(v):
    labels = [v[:2] + '0' * (len(v) - 2)]
    for i in range(2, len(v)):
        if v[i] == '0':
            break
        labels.append(v[:i + 1] + '0' * (len(v) - i - 1))
    return [l for l in labels if l in cpv]

df['label'] = df.cpv.progress_apply(cpv_to_label_hierarchy)

# Filter out cpv codes from descriptions
df.desc = df.desc.str.replace(r'\d{8}(-\d)?', '', regex=True)

# Hitin expects a list of tokens, so we make one
df['token'] = df.desc.progress_apply(lambda x: [x])

# Split data into train, val, test
import numpy as np

train, validate, test = np.split(df[['label', 'token']].sample(frac=1, random_state=42), [int(.7 * len(df)), int(.85 * len(df))])
with open('cpv/cpv_train.json', 'w', encoding='utf-8') as f:
    train.to_json(f, orient='records', lines=True, force_ascii=False)
with open('cpv/cpv_valu.json', 'w', encoding='utf-8') as f:
    validate.to_json(f, orient='records', lines=True, force_ascii=False)
with open('cpv/cpv_test.json', 'w', encoding='utf-8') as f:
    test.to_json(f, orient='records', lines=True, force_ascii=False)
