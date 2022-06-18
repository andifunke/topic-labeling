""" parsing arguments and setting default values """
import argparse

from topiclabeling.utils.constants import LOG_DIR


def global_args():
    parser = argparse.ArgumentParser(description="topic labeling project")

    # boolean
    parser.add_argument("--store", dest="store", action="store_true", required=False)
    parser.add_argument(
        "--no-store", dest="store", action="store_false", required=False
    )
    parser.set_defaults(store=True)

    # ints
    parser.add_argument("--start", type=int, required=False, default=0)
    parser.add_argument("--batch_size", type=int, required=False, default=None)
    parser.add_argument("--batches", type=int, required=False, default=1)

    # strings / paths
    parser.add_argument("--corpus", type=str, nargs="*", required=False, default="")
    parser.add_argument("--spacy_path", type=str, required=False, default="de")
    parser.add_argument(
        "--log",
        type=str,
        nargs="*",
        required=False,
        default=["stdout", "file"],
        choices=["stdout", "file", "none"],
    )  # TODO: add exclusivity for 'none'
    parser.add_argument("--log_path", type=str, required=False, default=LOG_DIR)
    parser.add_argument(
        "--lemmata",
        type=str,
        required=False,
        help="Path to a tab separated text file that maps surface forms to "
        "a normalized form. Can also be used to define exceptions for "
        "tokens that should not be lemmatized. An optional POS tag can "
        "can specified as an additional filter. Use the following pattern "
        "per row: TOKEN<tab>POS<tab>LEMMA",
    )

    args = parser.parse_args()

    return args
