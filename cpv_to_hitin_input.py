#{
#    "doc_label": ["Computer", "MachineLearning", "DeepLearning", "Neuro", "ComputationalNeuro"],
#    "doc_token": ["I", "love", "deep", "learning"],
#}



# https://simap.ted.europa.eu/en_GB/web/simap/cpv
# The first two digits identify the divisions (XX000000-Y);
# The first three digits identify the groups (XXX00000-Y);
# The first four digits identify the classes (XXXX0000-Y);
# The first five digits identify the categories (XXXXX000-Y);
# "92", "1", "1" "1", "2"
import pandas as pd
import csv

pd.set_option('display.max_colwidth', None)

df = pd.read_csv('cpv/all.csv', sep=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, usecols=[1, 2, 3])

df.dropna(inplace=True)

def int_to_hierarchy(int_cpv):
    v = str(int(int_cpv))
    return [subcategory for subcategory in [v[:2], *v[2:]] if subcategory != '0']

df.cpv = df.cpv.apply(int_to_hierarchy)

df.desc = df.desc.str.replace(r'\d{8}(-\d)?', '', regex=True)
print(df)
