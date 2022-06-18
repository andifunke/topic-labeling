import gc
import re

import pandas as pd

from topiclabeling.utils.constants import ETL_DIR, DATASETS_FULL, POS_N
from topiclabeling.utils.logging import logg
from topiclabeling.utils.utils import multiload


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


def generate_and_save_map(df, dataset):
    df = df.to_frame().rename(columns={"text": "counts"}).reset_index()
    df = df.groupby("token").apply(get_best_text)
    file = f"{DATASETS_FULL.get(dataset, dataset)}_lemmatization_map.pickle"
    logg(f"Writing {file}")
    df.to_pickle(ETL_DIR / file)
    gc.collect()


def main():
    # the following regex are mainly there in order to reduce the vocabulary-size of the dewac
    # corpus
    digits = r'[0-9.,/=:;&#\!\?\*"\'\-\(\)\[\]]+'
    web = r".*?(http:|www\.|\.html|\.htm|\.php|\.de|\.net|\.com|\.at|\.org|\.info).*"
    start = r'[/&\-ï",\'\$\(\)\*\.]+.*'
    end = r".[\.\(\)¬]*"
    badasc = r".*?[].*"
    pat = re.compile(
        r"^(" + "|".join([digits, web, start, end, badasc]) + r")$", flags=re.IGNORECASE
    )

    datasets = ["dewac", "dewiki"]
    for dataset in datasets[:]:
        series = []
        for df in multiload(dataset, "nlp"):
            gc.collect()
            df = df[df.POS.isin(POS_N)]
            df = df[~df.token.str.match(pat)]
            df = df.set_index("token").text
            series.append(df)
            gc.collect()
        df = pd.concat(series)
        df = df.groupby("token").value_counts()
        df = df[df > 1]
        generate_and_save_map(df, dataset)


if __name__ == "__main__":
    main()
