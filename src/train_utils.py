# coding: utf-8
import multiprocessing as mp
from os import makedirs
from os.path import join, exists

from gensim.models.callbacks import CallbackAny2Vec
import argparse


class EpochLogger(CallbackAny2Vec):
    """
    Callback to log information about training.
    Not serializable -> remove before saving the model.
    """

    def __init__(self, logger):
        self.epoch = 1
        self.logger = logger

    def on_epoch_begin(self, model):
        self.logger.info("Epoch #{:02d} start".format(self.epoch))

    def on_epoch_end(self, model):
        self.logger.info("Epoch #{:02d} end".format(self.epoch))
        self.epoch += 1


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after each epoch."""

    def __init__(self, model_name, directory, checkpoint_every=5):
        self.model_name = model_name
        self.directory = join(directory, "checkpoints")
        if not exists(self.directory):
            makedirs(self.directory)
        self.epoch = 1
        self.checkpoint_every = checkpoint_every

    def on_epoch_end(self, model):
        if self.epoch % self.checkpoint_every == 0:
            file = "{}_epoch{:02d}".format(self.model_name, self.epoch)
            filepath = join(self.directory, file)
            print("Saving checkpoint to " + filepath)
            callbacks = model.callbacks
            model.callbacks = ()
            model.save(filepath)
            model.callbacks = callbacks
        self.epoch += 1


def parse_args(default_model_name="x2v", default_epochs=20):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cacheinmem", dest="cache_in_memory", action="store_true", required=False
    )
    parser.add_argument(
        "--no-cacheinmem", dest="cache_in_memory", action="store_false", required=False
    )
    parser.set_defaults(cache_in_memory=False)
    parser.add_argument(
        "--lowercase", dest="lowercase", action="store_true", required=False
    )
    parser.add_argument(
        "--no-lowercase", dest="lowercase", action="store_false", required=False
    )
    parser.set_defaults(lowercase=False)
    parser.add_argument(
        "--fasttext", dest="fasttext", action="store_true", required=False
    )
    parser.add_argument(
        "--no-fasttext", dest="fasttext", action="store_false", required=False
    )
    parser.set_defaults(lowercase=False)

    parser.add_argument(
        "--model_name", type=str, required=False, default=default_model_name
    )
    parser.add_argument("--epochs", type=int, required=False, default=default_epochs)
    parser.add_argument("--min_count", type=int, required=False, default=20)
    parser.add_argument("--cores", type=int, required=False, default=mp.cpu_count())
    parser.add_argument("--checkpoint_every", type=int, required=False, default=10)

    args = parser.parse_args()
    return (
        args.model_name,
        args.epochs,
        args.min_count,
        args.cores,
        args.checkpoint_every,
        args.cache_in_memory,
        args.lowercase,
        args.fasttext,
        args,
    )
