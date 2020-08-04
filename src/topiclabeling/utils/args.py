""" parsing arguments and setting default values """
import argparse

from topiclabeling.utils.constants import LOG_DIR


def global_args():
    parser = argparse.ArgumentParser(description='topic labeling project')

    # boolean
    parser.add_argument('--store', dest='store', action='store_true', required=False)
    parser.add_argument('--no-store', dest='store', action='store_false', required=False)
    parser.set_defaults(store=True)

    # ints
    parser.add_argument('--start', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--batches', type=int, required=False)

    # strings / paths
    parser.add_argument('--corpus', type=str, nargs='*', required=False)
    parser.add_argument('--spacy_path', type=str, required=False, default='de')
    parser.add_argument('--log', type=str, nargs='*', required=False, default=['file'],
                        choices=['stdout', 'file', 'none'])  # TODO: add exclusivity for 'none'
    parser.add_argument('--log_path', type=str, required=False, default=LOG_DIR)

    args = parser.parse_args()

    return args
