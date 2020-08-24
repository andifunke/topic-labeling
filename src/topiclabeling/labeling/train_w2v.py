# coding: utf-8
import gc
from pathlib import Path
from time import time

import pandas as pd
from gensim.models import Word2Vec, FastText
from tqdm import tqdm

from topiclabeling.utils.constants import TOKEN, HASH, SENT_IDX, EMB_DIR
from topiclabeling.utils.logging import init_logging, log_args, logg
from topiclabeling.utils.train_utils import parse_args, EpochSaver, EpochLogger


class Sentences(object):
    """Memory friendly data streaming from disk."""

    def __init__(self, input_path, lowercase=False, input_format='text', use_file_cache=False):
        self.input_path = Path(input_path)
        if self.input_path.is_dir():
            self.files = sorted(f for f in input_path.iterdir() if f.is_file())
        else:
            self.files = [self.input_path]

        self.lowercase = lowercase
        self.pickle_format = input_format == 'pickle'

        if self.pickle_format:
            self.use_file_cache = use_file_cache
            self.cached_files = None
            if self.use_file_cache:
                self._init_file_cache()
            self.__iter__ = self._iter_pickle
        else:
            self.__iter__ = self._iter_text

    def __iter__(self):
        yield from self._iter_pickle() if self.pickle_format else self._iter_text()

    def _iter_text(self):
        for filename in self.files:
            with open(filename) as fp:
                logg(f"Reading {filename}")
                yield from map(lambda x: x.strip().split(), fp)

    def _iter_pickle(self):
        if self.use_file_cache:
            for f in self.cached_files:
                gc.collect()
                logg(f)
                ser = pd.read_pickle(f)
                for sent in ser:
                    yield sent
        else:
            for filename in self.files:
                ser = self._load(filename)
                for sent in ser:
                    yield sent

    def _init_file_cache(self):
        cache_dir = self.input_path / 'train_cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        logg(f"initializing file cache in {cache_dir}")

        for file in self.files:
            ser = self._load(file)
            ser.to_pickle(cache_dir / f"{file.name.split('.')[0]}_cache.pickle")

        self.cached_files = sorted(f for f in cache_dir.iterdir() if f.is_file())

    def _load(self, file_path):
        gc.collect()
        logg(file_path)
        df = pd.read_pickle(file_path)

        if self.lowercase:
            df.token = df.token.str.lower()

        df = df.groupby([HASH, SENT_IDX], sort=False)[TOKEN].agg(self.docs_to_lists)

        return df

    @staticmethod
    def docs_to_lists(token_series):
        return tuple(token_series.tolist())


def main():
    t0 = time()

    # --- argument parsing ---
    args = parse_args(default_model_name='w2v', default_epochs=100)

    # --- init logging ---
    init_logging(name=args.model_name, to_stdout='stdout' in args.log, to_file='file' in args.log)
    log_args(args)

    # --- setting up paths ---
    input_path = Path(args.input)
    model_path = EMB_DIR / args.model_name
    model_path.mkdir(exist_ok=True, parents=True)
    logg(f"model dir: {model_path}")

    # --- loading data ---
    sentences = Sentences(
        input_path, lowercase=args.lowercase, input_format=args.format, use_file_cache=args.stream,
    )
    if not args.stream:
        # needs approx. 25GB of RAM for Wikipedia
        logg("caching data in memory")
        sentences = [s for s in tqdm(sentences, unit=' lines')]
        gc.collect()

    # Model initialization
    logg("Initializing new model")
    train_params = dict(
        size=300,
        window=5,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
        sample=1e-5,
        negative=5,
        sg=1,
        seed=42,
        iter=args.epochs,
        workers=args.cores,
    )
    if args.fasttext:
        model = FastText(**train_params, min_n=3, max_n=6)
    else:
        model = Word2Vec(**train_params)

    logg("Building vocab")
    model.build_vocab(sentences, progress_per=100_000)

    # Model training
    epoch_saver = EpochSaver(model_path, args.checkpoint_every)
    epoch_logger = EpochLogger()

    logg(f"Training {args.epochs:d} epochs")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs,
        report_delay=60,
        callbacks=[epoch_logger, epoch_saver],
    )

    # saving model
    file_path = model_path / args.model_name
    logg(f"Writing model to {file_path}")
    model.callbacks = ()
    model.save(str(file_path))

    t1 = int(time() - t0)
    logg(f"all done in {t1//3600:02d}:{(t1//60) % 60:02d}:{t1 % 60:02d}")


if __name__ == '__main__':
    main()
