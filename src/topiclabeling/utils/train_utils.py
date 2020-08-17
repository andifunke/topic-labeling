# coding: utf-8
import argparse
import multiprocessing as mp
from pathlib import Path

from gensim.models.callbacks import CallbackAny2Vec

from topiclabeling.utils.logging import logg


class EpochLogger(CallbackAny2Vec):
    """
    Callback to log information about training.

    Not serializable -> remove before saving the model.
    """

    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        logg(f"Epoch #{self.epoch:02d} start")

    def on_epoch_end(self, model):
        logg(f"Epoch #{self.epoch:02d} end")
        self.epoch += 1


class EpochSaver(CallbackAny2Vec):
    """
    Callback to save model after each epoch.

    For compatibility -> remove before saving the model.
    """

    def __init__(self, model_path, checkpoint_every=5):
        model_path = Path(model_path)
        self.model_name = model_path.name
        self.directory = model_path / 'checkpoints'
        self.directory.mkdir(exist_ok=True, parents=True)
        self.checkpoint_every = checkpoint_every
        self.epoch = 1

    def on_epoch_end(self, model):
        if self.epoch % self.checkpoint_every == 0:
            file = f'{self.model_name}_epoch{self.epoch:02d}'
            file_path = self.directory / file
            print(f'Saving checkpoint to {file_path}')
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

    parser.add_argument("--input", type=str, required=False, default='dewiki',
                        help="Full path to an input corpus or common corpus name.")
    parser.add_argument("--format", type=str, required=False, choices=['text', 'pickle'],
                        default='text',
                        help="text: expects a text file with one sentence/document per line. "
                             "pickle: expects a pickled dataframe in the common package format.")
    parser.add_argument("--model_name", type=str, required=False, default=default_model_name)
    parser.add_argument("--epochs", type=int, required=False, default=default_epochs)
    parser.add_argument("--min_count", type=int, required=False, default=20)
    parser.add_argument("--cores", type=int, required=False, default=mp.cpu_count())
    parser.add_argument("--checkpoint_every", type=int, required=False, default=10)
    parser.add_argument('--log', type=str, nargs='*', required=False, default=['stdout', 'file'],
                        choices=['stdout', 'file', 'none'])  # TODO: add exclusivity for 'none'

    args = parser.parse_args()

    return args
