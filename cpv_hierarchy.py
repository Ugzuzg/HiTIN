import pandas as pd
import json

def build_tree(df):
    def find_parent(row):
        """A parent of a CPV code is the closest matching CPV code with some values zeroed out.

        03300000 is a parent of
        03310000

        03300000 can also be a parent of
        03341000 if 03340000 does not exist
        """
        for i in range(row['level'], 0, -1):
            suspected_parent = (row['CODE'][:(i+1)] + '0' * (7 - i))
            parent = df.loc[df['CODE'] == suspected_parent, 'CODE']
            if not parent.empty:
                return parent.item()
        return None

    df['level'] = df['CODE'].apply(get_cpv_level)
    df['parent'] = df.apply(find_parent, axis=1)

    # df = df[df['level'] < 2]

    def _build_tree(parent=None):
        children = df[df['parent'].isna()] if parent is None else df[df['parent'] == parent]
        return [{ 'cpv': node['CODE'], 'name': node['name'], 'children': _build_tree(node['CODE']) } for _, node in children.iterrows()]

    return _build_tree()

def get_cpv_level(cpv_code: str):
    """A level of a CPV code is the number of non zero digits in the code after the first two digits."""
    return len(cpv_code[2:]) - cpv_code[2:].count('0')

# Read 'CODE' as object, otherwise pandas converts it to integers removing leading zeros
cpv = pd.read_csv('cpv/cpv.csv', dtype={'CODE': object})
cpv_tree = build_tree(cpv)

with open('cpv/cpv_hierarchy.json', 'w') as f:
    json.dump(cpv_tree, f, indent=2)
