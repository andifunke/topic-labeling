""" parsing arguments and setting default values """
import argparse


def get_options():
    parser = argparse.ArgumentParser(description='topic labeling project')

    # boolean
    parser.add_argument('--hpc', dest='hpc', action='store_true')
    parser.add_argument('--no-hpc', dest='hpc', action='store_false')
    parser.set_defaults(hpc=False)
    parser.add_argument('--log', dest='log', action='store_true')
    parser.add_argument('--no-log', dest='log', action='store_false')
    parser.set_defaults(log=True)

    # strings / paths
    parser.add_argument('--spacy_model_path', default='de', type=str)
    parser.add_argument('--log_path', default='./../logs/nlp_pipe.log', type=str)
    parser.add_argument('--corpus_prefix', default='', nargs='*')

    # dummy argument for IPython
    parser.add_argument('-f', '--file', help='dummy argument for IPython')

    # build and modify argument dictionary
    options = vars(parser.parse_args())
    if options['hpc'] and options['spacy_model_path'] == 'de':
        options['spacy_model_path'] = \
            '/home/funkea/.local/lib/python3.4/site-packages/de_core_news_sm/de_core_news_sm-2.0.0'

    return options
