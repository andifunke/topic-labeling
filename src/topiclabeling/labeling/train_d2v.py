# coding: utf-8
import gc
from pathlib import Path
from time import time

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import RULE_KEEP, RULE_DISCARD
from tqdm import tqdm

from topiclabeling.utils.constants import ETL_DIR, HASH, TOKEN, TEXT, EMB_DIR
from topiclabeling.utils.logging import logg, init_logging, log_args
from topiclabeling.utils.train_utils import parse_args, EpochSaver, EpochLogger, \
    SynonymJudgementTaskDEMetric


class Documents(object):

    def __init__(self, input_path, lowercase=False, input_format='text'):
        self.input_path = Path(input_path)
        if self.input_path.is_dir():
            self.files = sorted(f for f in input_path.iterdir() if f.is_file())
        else:
            self.files = [self.input_path]

        self.lowercase = lowercase
        self.pickle_format = input_format == 'pickle'

    def __iter__(self):
        yield from self._iter_pickle() if self.pickle_format else self._iter_text()

    def _iter_text(self):

        i = 0

        def tag(doc):
            nonlocal i
            i += 1
            tagged_doc = TaggedDocument(doc.strip().split(), [i])
            return tagged_doc

        for filename in self.files:
            with open(filename) as fp:
                logg(f"Reading {filename}")
                yield from map(tag, tqdm(fp, unit=' documents'))

    def _iter_pickle(self):
        titles = pd.read_pickle(ETL_DIR / 'dewiki_phrases_lemmatized.pickle')
        if self.lowercase:
            titles.token = titles.token.str.lower()
            titles.text = titles.text.str.lower()

        for file in self.files:
            gc.collect()
            corpus = file.name.split('.')[0]
            logg(f'loading {corpus}')

            df = pd.read_pickle(file)

            # applying the same processing to each document on each iteration 
            # is rather inefficient. If applicable keep TaggedDocuments in memory
            if self.lowercase:
                df.token = df.token.str.lower()
            df = df.groupby([HASH], sort=False)[TOKEN].agg(self.docs_to_lists)

            for doc_id, doc in df.iteritems():
                # The conversion of the hash_id to str is necessary since gensim tries to
                # allocate an array for ids of size 2^64 if int values are too big. 2nd tag
                # is the lemmatized token, 3rd tag is the original (underscore-concatenated)
                # title (parenthesis removed)
                yield TaggedDocument(doc, [
                    str(doc_id),
                    titles.loc[doc_id, TOKEN],
                    titles.loc[doc_id, TEXT]
                ])

    @staticmethod
    def docs_to_lists(token_series):
        return tuple(token_series.tolist())


# TODO: merge with `train_w2v`
def main():
    t0 = time()

    # --- argument parsing ---
    args = parse_args(default_model_name='d2v', default_epochs=50)

    # --- init logging ---
    init_logging(name=args.model_name, to_stdout='stdout' in args.log, to_file='file' in args.log)
    log_args(args)

    input_path = Path(args.input)
    model_path = EMB_DIR / args.model_name
    model_path.mkdir(exist_ok=True, parents=True)
    logg(f"model dir: {model_path}")

    documents = Documents(input_path, lowercase=args.lowercase, input_format=args.format)
    if not args.stream:
        # needs approx. 25GB of RAM for Wikipedia
        logg("caching data in memory")
        documents = [s for s in tqdm(documents, unit=' lines')]
        gc.collect()

    # Model initialization
    if args.from_checkpoint is None:
        logg("Initializing new model")
        model = Doc2Vec(
            vector_size=300,
            window=15,
            min_count=args.min_count,
            max_vocab_size=args.max_vocab_size,
            sample=1e-5,
            negative=5,
            hs=0,
            dm=0,
            dbow_words=1,
            dm_concat=0,
            seed=42,
            epochs=args.epochs,
            workers=args.cores,
        )
        logg("Building vocab")
        if args.vocab:
            with open(args.vocab) as fp:
                print(f'Loading vocab file {args.vocab}')
                vocab_terms = sorted({line.strip() for line in fp.readlines()})
                print(f'{len(vocab_terms)} terms loaded.')
        else:
            vocab_terms = []

        def trim_rule(word, count, min_count):
            if word in vocab_terms:
                return RULE_KEEP
            if count >= min_count:
                return RULE_KEEP
            return RULE_DISCARD

        model.build_vocab(documents, trim_rule=trim_rule)
    else:
        if not Path(args.from_checkpoint).exists():
            raise ValueError(f"Path {args.from_checkpoint} does not exists")
        logg(f'Loading model from {args.from_checkpoint}')
        model = Doc2Vec.load(args.from_checkpoint)

    # Model training
    epoch_saver = EpochSaver(model_path, args.checkpoint_every, start_epoch=args.from_epoch)
    epoch_logger = EpochLogger(start_epoch=args.from_epoch)
    sjt_de = SynonymJudgementTaskDEMetric(call_every=1, start_epoch=args.from_epoch)

    logg(f"Training {args.epochs:d} epochs")
    model.train(
        documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
        report_delay=60,
        callbacks=[epoch_logger, sjt_de, epoch_saver],
    )

    # saving model
    file_path = model_path / args.model_name
    logg(f"Writing model to {file_path}")
    model.callbacks = ()
    model.save(str(file_path))

    t1 = int(time() - t0)
    logg(f"all done in {t1 // 3600:02d}:{(t1 // 60) % 60:02d}:{t1 % 60:02d}")


if __name__ == '__main__':
    main()
