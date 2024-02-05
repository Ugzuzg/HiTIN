import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

# URL to download CPV codes dataset in ZIP format (flat hierarchy)
URL_CPV_CODES = "https://ted.europa.eu/documents/d/ted/cpv_2008_xls"

# Fetch CPV codes from the SIMAP website with their names in a given language
def get_cpv_codes_from_simap(language):
    zipfile = ZipFile(BytesIO(urlopen(URL_CPV_CODES).read()))
    data = pd.read_excel(zipfile.open(zipfile.namelist()[0]))
    
    # Rename the language column to 'name', 
    # because xls file contains names in different languages and columns are named after each language
    data.rename(columns={language: 'name'}, inplace=True)
    
    return data[['CODE', 'name']]

# Fetch CPV codes for English language 
# and truncate them to the first 8 characters to remove error correction code, which goes after '-'
df = get_cpv_codes_from_simap('EN')
df['CODE'] = df['CODE'].apply(lambda x: x[:8])

df.to_csv('cpv/cpv.csv', index=False)
