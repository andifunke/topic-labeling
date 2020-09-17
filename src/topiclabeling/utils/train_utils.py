# coding: utf-8
import argparse
import multiprocessing as mp
from math import nan
from pathlib import Path

import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec

from topiclabeling.utils.constants import TMP_DIR
from topiclabeling.utils.logging import logg


class EpochLogger(CallbackAny2Vec):
    """
    Callback to log information about training.

    Not serializable -> remove before saving the model.
    """

    def __init__(self, start_epoch: int = 1):
        self.epoch = start_epoch

    def on_epoch_begin(self, model):
        logg(f"Epoch #{self.epoch:02d} start")

    def on_epoch_end(self, model):
        logg(f"Epoch #{self.epoch:02d} end")
        try:
            logg(f"Train loss: {model.get_latest_training_loss()}")
        except AttributeError:
            pass
        self.epoch += 1


# TODO: add EN metric
class SynonymJudgementTaskDEMetric(CallbackAny2Vec):
    """Perform a German Synonym Judgement Task at the end of each epoch."""

    def __init__(self, call_every: int = 1, start_epoch: int = 1):
        sj_file_de = TMP_DIR / 'synonym_judgement/SJT_stimuli.csv'
        sj_de_full = pd.read_csv(sj_file_de)
        sj_de = sj_de_full[['probe', 'target', 'foil1', 'foil2']]
        self.sj_de = sj_de[~sj_de.isna().any(axis=1)]
        self.call_every = call_every
        self.epoch = start_epoch

    @staticmethod
    def closest_match(terms, vectors):
        """
        Returns the index of the term closest to the first term in a list of words.

        Note that index 0 is taken as the probe and all words with index > 0 are tested.
        """

        terms = terms.to_list()
        try:
            distances = vectors.distances(terms[0], terms[1:])
            min_dist = distances.argmin() + 1
            return min_dist
        except KeyError:
            return -1

    def synonym_judgement_accuracy(self, word_vectors, target_idx=1):
        pred = self.sj_de.apply(lambda x: self.closest_match(x, word_vectors), axis=1)
        pred = pred[pred > 0]
        correct = (pred == target_idx).sum()
        acc = correct / len(pred) if len(pred) else nan
        logg(f"SJT accuracy: {round(acc, 3)}")
        logg(f"Number of tests omitted due to OOV terms: {len(self.sj_de) - len(pred)}")

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        if self.epoch % self.call_every == 0:
            self.synonym_judgement_accuracy(model.wv)
        self.epoch += 1


class EpochSaver(CallbackAny2Vec):
    """
    Callback to save model after each epoch.

    For compatibility -> remove before saving the model.
    """

    def __init__(self, model_path, checkpoint_every: int = 5, start_epoch: int = 1):
        model_path = Path(model_path)
        self.model_name = model_path.name
        self.directory = model_path / 'checkpoints'
        self.directory.mkdir(exist_ok=True, parents=True)
        self.checkpoint_every = checkpoint_every
        self.epoch = start_epoch

    def on_epoch_end(self, model):
        if self.epoch % self.checkpoint_every == 0:
            file = f'{self.model_name}_epoch{self.epoch:02d}'
            file_path = self.directory / file
            logg(f'Saving checkpoint to {file_path}')
            callbacks = model.callbacks
            model.callbacks = ()
            model.save(str(file_path))
            model.callbacks = callbacks

        self.epoch += 1


def parse_args(default_model_name='x2v', default_epochs=20):
    parser = argparse.ArgumentParser()

    parser.add_argument('--stream', dest='stream', action='store_true', required=False)
    parser.add_argument('--no-stream', dest='stream', action='store_false', required=False)
    parser.set_defaults(stream=False)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true', required=False)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false', required=False)
    parser.set_defaults(lowercase=False)
    parser.add_argument('--fasttext', dest='fasttext', action='store_true', required=False)
    parser.add_argument('--no-fasttext', dest='fasttext', action='store_false', required=False)
    parser.set_defaults(fasttext=False)

    parser.add_argument("--input", type=str, default='dewiki',
                        help="Full path to an input corpus or common corpus name.")
    parser.add_argument("--format", type=str, default='text', choices=['text', 'pickle'],
                        help="text: expects a text file with one sentence/document per line. "
                             "pickle: expects a pickled dataframe in the common package format.")
    parser.add_argument("--model_name", type=str, default=default_model_name)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--min_count", type=int, default=20)
    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--cores", type=int, default=mp.cpu_count())
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument('--log', type=str, nargs='*', default=['stdout', 'file'],
                        choices=['stdout', 'file', 'none'])  # TODO: add exclusivity for 'none'
    parser.add_argument('--vocab', type=str, required=False,
                        help="File path containing terms per line to be included in the "
                             "model's vocabulary. "
                             "Terms will only be in the vocab, if found in the corpus.")
    parser.add_argument("--from_checkpoint", type=str, default=None,
                        help="Load checkpoint from path and continue training.")
    parser.add_argument("--from_epoch", type=int, default=1,
                        help="Specify an offset index for the epochs in order to not overwrite "
                             "existing checkpoints. Otherwise the training will start with "
                             "epoch 1, even when training continues from a loaded checkpoint.")

    args = parser.parse_args()

    return args
