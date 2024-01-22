import pandas as pd
import json

def build_tree(df):
    def find_parent(row):
        for i in range(row['level'], 0, -1):
            suspected_parent = (row['CODE'][:(i+1)] + '0' * (7 - i))
            parent = df.loc[df['CODE'] == suspected_parent, 'CODE']
            if not parent.empty:
                return parent.item()
        return None

    df['level'] = df['CODE'].apply(get_cpv_level)
    for i in range(6, -1, -1):
        level = df[df['level'] == i]
        df.loc[level.index, 'parent'] = level.apply(find_parent, axis=1)

    # df = df[df['level'] < 2]

    def _build_tree(parent=None):
        children = df[df['parent'].isna()] if parent is None else df[df['parent'] == parent]
        return [{ 'cpv': node['CODE'], 'name': node['name'], 'children': _build_tree(node['CODE']) } for _, node in children.iterrows()]

    return _build_tree()

def get_cpv_level(cpv_code):
    return 6 - cpv_code.split('-')[0][2:].count('0')

cpv = pd.read_csv('cpv/cpv.csv', dtype={'CODE': object})
with open('cpv/cpv_hierarchy.json', 'w') as f:
    json.dump(build_tree(cpv), f, indent=2)
