# coding: utf-8

from os.path import join
import re
from time import time
import pandas as pd
import csv
import locale; locale.setlocale(locale.LC_ALL, '')
import xml.etree.ElementTree as et
from html import unescape
from datetime import datetime

from constants import DATA_BASE, ETL_PATH, \
    META, DATASET, SUBSET, ID, ID2, TITLE, TAGS, TIME, DESCRIPTION, TEXT, LINKS, DATA, HASH
from utils import hms_string

CORPUS = "dewiki"
LOCAL_PATH = "dewiki/dewiki-latest-pages-articles.xml"
IN_PATH = join(DATA_BASE, LOCAL_PATH)
OUT_PATH = join(ETL_PATH, CORPUS)


def strip_tag(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag


def split_title(title):
    split = title.find('(', 1)
    split = None if split < 1 else split
    return title[:split], (title[split:] if split else '')


def store(rows, fname):
    df = pd.DataFrame.from_dict(rows)
    print('saving to', fname)
    df = df.set_index(HASH)[META+DATA]
    df.to_pickle(fname)


def parse_xml(infile, outfile, iterations, batch=None, print_every=1000):
    """
    :param infile: path to Wikipedia xml dump
    :param outfile: path to writable csv file
    :param iterations: number all articles to process and write. None for all.
    :param batch: append to csv file every m articles.
                        if None: writes a pickled DataFrame *after* processing the entire corpus
                                disadvantage: high memory consumption
                                advantage: keeps objects as byte streams
    :param print_every: print progress to stdout every n articles
    :return:
    """
    outfile += '.csv' if batch else '.pickle'

    with open(infile, 'r') as fr, open(outfile, 'w') as fw:

        fields = [HASH] + META + DATA
        writer = csv.DictWriter(fw, fields, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        article_count = 0
        rows = []
        row = None
        is_article = is_redirect = False

        for event, elem in et.iterparse(fr, events=['start', 'end']):
            tag = strip_tag(elem.tag)

            if event == 'start':
                # start new row
                if tag == 'page':
                    row = dict()
            else:
                if tag == 'title':
                    row[TITLE], row[DESCRIPTION] = split_title(elem.text)
                elif tag == 'ns' and int(elem.text) == 0:
                    is_article = True
                    row[DATASET] = CORPUS
                    row[SUBSET] = ""
                elif tag == 'id' and tag not in row:
                    row[ID] = elem.text
                    row[ID2] = 0
                elif tag == 'timestamp':
                    row[TIME] = datetime.strptime(elem.text.replace('Z', 'UTC'),
                                                  '%Y-%m-%dT%H:%M:%S%Z')  # 2018-07-29T18:22:20Z
                elif tag == 'redirect':
                    is_redirect = True
                    row[LINKS] = elem.get('title')
                elif tag == 'text' and is_article:
                    if not is_redirect:
                        row[TEXT], row[TAGS], row[LINKS] = parse_markdown(elem.text)
                        row[TAGS] = tuple(row[TAGS])
                        # dump empty pages
                        if not row[TEXT]:
                            is_article = False
                    else:
                        row[TEXT], row[TAGS], row[LINKS] = "", tuple(), list()
                # write and close row, reset flags
                elif tag == 'page':
                    if is_article:
                        row[HASH] = hash(tuple([row[key] for key in META]))
                        article_count += 1
                        rows.append(row)
                        # print status
                        if article_count > 1:
                            if (article_count % print_every) == 0:
                                print(locale.format("%d", article_count, grouping=True))
                            if batch:
                                # if batch is False we will save everything *after* processing
                                if (article_count % batch) == 0:
                                    # write batch of rows and reset list of rows
                                    writer.writerows(rows)
                                    rows = []
                    # reset everything
                    row = None
                    is_article = is_redirect = False
                elem.clear()

            if article_count == iterations:
                break

    if not batch:
        store(rows, outfile)


# matches against: [[Kategorie:Soziologische Systemtheorie]], [[Kategorie:Fiktive Person|Smithee, Alan]]
category = r"\[\[Kategorie:(?P<cat>[\w ]+)(?:\|.*)?\]\]"
re_category = re.compile(category)

# remove tags
refs = r"<\s*(ref|math)[^>.]*?(?:\/\s*>|>.*?<\s*\/\s*(ref|math)\s*>)"
tags = r"<(.|\n)*?>"
re_tags = re.compile(r"(%s|%s)" % (refs, tags), re.MULTILINE)

table = r"{\|(?s:.*?)\|}"
re_table = re.compile(table)

# remove meta data: [[Datei:...]]
chars = r"\xa0"
emph = r"\'{2,}"
bullet = r"^[\*:] *"
bullet2 = r"^\|.*"
meta = r"\[\[\w+:.*?\]\]"
footer = r"== (Bibliographie|Literatur|Weblinks|Einzelnachweise) ==(?s:.)*"
re_meta = re.compile(r"(%s|%s|%s|%s|%s|%s)" % (chars, emph, bullet, bullet2, meta, footer), re.MULTILINE)

# => merge ^
remove = r'(' + r'|'.join([refs, tags, table, chars, emph, bullet, bullet2, meta, footer]) + r')'
re_remove = re.compile(remove, re.MULTILINE)

# matches against: [[Aristoteles]], [[Reductio ad absurdum|indirekten Beweis]]
wikilink = r"\[\[(.*?)\]\]"
re_link = re.compile(wikilink)

zitat = r"{{Zitat(?:\||-.*?\|Übersetzung=)(?P<token>.*?)(?:\|.*?)?}}"
re_zitat = re.compile(zitat)

replace = r'(' + r'|'.join([wikilink, zitat]) + r')'
re_replace = re.compile(replace)

infobox = r"{{.*?(?:}}|(?={{))"
re_infobox = re.compile(infobox, re.DOTALL)

lf = r"\n\n*\n(?!==)"
re_lf = re.compile(lf)


def parse_markdown(text):
    """
    This is actucally not a real parser since it keeps no internal states. Therefore nested structures
    are a bit of a problem and a few artifacty may remain. Also the regexes are a bit nasty and need to
    read the text multiple times. Maybe I'm doing a new version at some point, but for the time being
    it's working sufficiently well.
    """
    # replace html escapings
    text = unescape(text)

    # extract categories
    categories = re_category.findall(text)

    # remove formatting tags and ref/math tags with content
    # This regex is actually working way better than the w3lib.remove_tags[_with_content] implementations.
    # It's ~1.5x faster and keeps all wanted content, while the w3lib methods introduce problems with some
    # self-closing xml-tags. Of course lxml/beautifulsoup would be another option.
    text = re_tags.sub('', text)

    # remove tables
    text = re_table.sub('', text)

    # remove metadata and formatting
    text = re_meta.sub('', text)

    # replace citations
    text = re_zitat.sub(r"„\g<token>”", text)

    # replace WikiLinks
    links = []

    def replace_links(matchobj):
        split = matchobj.group(1).split('|', 1)
        links.append(split[0])
        return split[1] if len(split) > 1 else split[0]

    text = re_link.sub(replace_links, text)

    # repeat for nested structures, performancewise not perfect
    n = 1
    while n > 0:
        text, n = re_infobox.subn('', text)

    text = re_lf.sub('\n', text)
    return text.strip(' \n}{'), categories, links


if __name__ == '__main__':
    t0 = time()
    parse_xml(IN_PATH, OUT_PATH, iterations=None, batch=0, print_every=10000)
    t1 = int(time() - t0)
    print("all done in", hms_string(t1))
