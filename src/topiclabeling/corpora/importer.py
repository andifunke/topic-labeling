import json
import hashlib
from pathlib import Path
from typing import Iterable, Any

import pandas as pd
from datetime import datetime
import re
import gzip
from bs4 import BeautifulSoup
from html import unescape

from topiclabeling.utils.constants import (
    DATA_DIR,
    ETL_DIR,
    DATASET,
    SUBSET,
    ID,
    ID2,
    TITLE,
    TIME,
    META,
    TEXT,
    DESCRIPTION,
    LINKS,
    TAGS,
    DATA,
    HASH,
    PathLike,
)


class CorpusImporter:
    @staticmethod
    def write_dataframe(corpus, df):
        """Returns the file name where the dataframe was stores."""

        ETL_DIR.mkdir(exist_ok=True, parents=True)
        file_name = ETL_DIR / corpus + ".csv"
        print(f"saving to {file_name}")
        df.to_csv(file_name)

        return file_name

    @staticmethod
    def hexhash(obj: Any) -> str:
        """Hashes a string and returns the MD5 hexadecimal hash as a string."""

        story_hash = hashlib.md5(str(obj).strip().encode("utf8"))
        hex_digest = story_hash.hexdigest()

        return hex_digest

    @staticmethod
    def write_document():
        pass


class OnlineParticipationImporter(CorpusImporter):

    CORPUS = "OnlineParticipation"
    LOCAL_PATH = "OnlineParticipationDatasets/downloads"

    def __init__(self, corpus_path: PathLike = None):

        self.corpus_path = (
            DATA_DIR / self.LOCAL_PATH if corpus_path is None else Path(corpus_path)
        )

    def transform_subset(self, source: Iterable[dict], subset_name: str):
        """
        :param source: list or iterator of dictionaries in original key/value format
        :param subset_name: string identifier of the subset the data belongs to

        :yields: dicts with normalized keys
        """
        category_lookup = {}
        print("transform", subset_name)

        for doc in source:
            if not doc["content"]:
                continue

            target = {
                DATASET: self.CORPUS,
                SUBSET: subset_name,
                ID: doc["suggestion_id"],
                TITLE: doc["title"],
                TIME: doc["date_time"],
                DESCRIPTION: None,
            }

            # 'wuppertal' has a different data scheme
            if subset_name == "wuppertal2017":
                if "tags" in doc:
                    target[TAGS] = tuple(doc["tags"])
                    category_lookup[target[ID]] = target[TAGS]
                else:
                    target[TAGS] = category_lookup[target[ID]]
                target[ID2] = None
                target[TEXT] = (
                    f"{doc['content']} .\n"
                    f"{doc['Voraussichtliche Rolle für die Stadt Wuppertal']} .\n"
                    f"{doc['Mehrwert der Idee für Wuppertal']} .\n"
                    # f"{doc['Eigene Rolle bei der Projektidee']} .\n"
                    # f"{doc['Geschätzte Umsetzungsdauer und Startschuss']} .\n"
                    # f"{doc['Kostenschätzung der Ideeneinreicher']} .\n"
                )
            else:
                if "category" in doc:
                    target[TAGS] = doc["category"]
                    category_lookup[target[ID]] = target[TAGS]
                else:
                    target[TAGS] = category_lookup[target[ID]]
                target[ID2] = doc["comment_id"] if ("comment_id" in doc) else 0
                target[LINKS] = target[ID] if target[ID2] else None
                target[TEXT] = doc["content"]

            target[HASH] = self.hexhash([target[key] for key in META])

            yield target

    def load_data(self, number_of_subsets: int = None, start: int = 0):
        """
        :param number_of_subsets: number of subsets to process in one call (None for no limit)
        :param start: index of first subset to process
        :yield: data set name, data subset name, data json
        """

        print(f"process {self.CORPUS}")

        # --- read files ---
        files = [f for f in self.corpus_path.iterdir() if f.is_file()]

        if number_of_subsets:
            number_of_subsets += start
            if number_of_subsets > len(files):
                number_of_subsets = None

        for file_path in files[start:number_of_subsets]:
            if file_path.name[-9:-5] != "flat":
                continue

            try:
                with open(file_path, "r") as fp:
                    print("open:", file_path)
                    data = json.load(fp)
                    if not data:
                        continue
            except IOError:
                print("Could not open", file_path)
                continue
            subset = file_path.name[6:-10]

            yield self.transform_subset(data, subset)

    def __call__(self):
        df = [pd.DataFrame(item) for item in self.load_data()]
        if df:
            df = pd.concat(df)
            df = df.set_index(HASH)[META + DATA]
        return df
