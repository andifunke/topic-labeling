import gc
import re
from os.path import join

import pandas as pd
from tqdm import tqdm

from constants import ETL_PATH, DSETS, POS_N
from utils import multiload

tqdm.pandas()
pd.options.display.max_rows = 2000


def get_best_text(grp):
    token: str = grp.name
    texts = grp.text.values
    counts = grp.counts.values
    # If there is only one option, return it (avoid unnecessary checks).
    if len(texts) == 1:
        return texts[0]
    # If the token is fully uppercase and contained in texts we assume it is a common abbreviation
    # or named entity like 'ABBA' which is supposed to be written uppercase: return the token.
    if token.isupper() and token in texts:
        return token
    # Else: remove all options less than max-count.
    texts = texts[counts == counts[0]]
    # Another shortcut to avoid unnecessary checks.
    if len(texts) == 1:
        return texts[0]
    # If the token is contained in the remaining options, return it.
    if token in texts:
        return token
    # Else: all remaining options are equally likely, return the first.
    return texts[0]


# the following regex are mainly there in order to reduce the vocabulary-size of the dewac corpus
digits = r'[0-9.,/=:;&#\!\?\*"\'\-\(\)\[\]]+'
web =    r'.*?(http:|www\.|\.html|\.htm|\.php|\.de|\.net|\.com|\.at|\.org|\.info).*'
start =  r'[/&\-ï",\'\$\(\)\*\.]+.*'
end =    r'.[\.\(\)¬]*'
badasc = r'.*?[].*'
pat = re.compile(
    r'^(' + '|'.join([
        digits,
        web,
        start,
        end,
        badasc
    ]) + r')$',
    flags=re.IGNORECASE
)


def generate_and_save_map(df, dataset):
    df = df.to_frame().rename(columns={'text': 'counts'}).reset_index()
    df = df.groupby('token').progress_apply(get_best_text)
    file = f'{DSETS.get(dataset, dataset)}_lemmatization_map.pickle'
    print(f'Writing {file}')
    df.to_pickle(join(ETL_PATH, file))
    gc.collect()


datasets = ['dewac', 'dewiki']
df = None
for dataset in datasets[1:]:
    series = []
    for df in multiload(dataset, 'nlp'):
        gc.collect()
        df = df[df.POS.isin(POS_N)]
        df = df[~df.token.str.match(pat)]
        df = df.set_index('token').text
        series.append(df)
        gc.collect()
    df = pd.concat(series)
    df = df.groupby('token').value_counts()
    df = df[df > 1]
    generate_and_save_map(df, dataset)
