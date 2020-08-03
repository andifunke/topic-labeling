# coding: utf-8

import locale
import re
import xml.etree.ElementTree as et
from datetime import datetime
from html import unescape
from os.path import join
from time import time

import pandas as pd

from topiclabeling.constants import (
    DATA_BASE, ETL_PATH, META, DATASET, SUBSET, ID, ID2, TITLE,
    TAGS, TIME, DESCRIPTION, TEXT, LINKS, DATA, HASH
)
from topiclabeling.utils import hms_string

locale.setlocale(locale.LC_ALL, '')

FIELDS = [HASH] + META + DATA
CORPUS = "dewiki"
LOCAL_PATH = "dewiki/dewiki-latest-pages-articles.xml"
IN_PATH = join(DATA_BASE, LOCAL_PATH)
OUT_PATH = join(ETL_PATH, CORPUS)


def strip_tag(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag


re_title = re.compile(r' \((.+?)\)$')


def split_title(title):
    matchobj = re.search(re_title, title)
    if matchobj:
        title = title[:matchobj.start(1)-2]
        category = matchobj[1]
    else:
        category = None
    return title, category


class DataFrameWriter(object):
    def __init__(self, outfile, fields):
        self.outfile = outfile
        self.fields = fields
        self.count = 0

    def write_rows(self, rows):
        if rows:
            df = pd.DataFrame(rows, columns=self.fields)
            df.set_index(HASH, drop=True, inplace=True)
            self.count += 1
            fname = "{}_{:02d}.pickle".format(self.outfile, self.count)
            print('saving to', fname)
            df.to_pickle(fname)


def parse_xml(infile, outfile, iterations, batch_size=100000, print_every=10000):
    """
    :param infile: path to Wikipedia xml dump
    :param outfile: path to writable file. .csv or .pickle will be appended.
    :param iterations: number all articles to process and write. None for all.
    :param batch_size: append to csv file or DataFrame every m articles
    :param print_every: print progress to stdout every n articles
    :return:
    """
    print("reading", infile)
    with open(infile, 'r') as fr:

        batch_writer = DataFrameWriter(outfile, FIELDS)

        # init counters, flags and data structures
        article_count = 0
        row = is_article = is_redirect = False
        rows = list()

        for event, elem in et.iterparse(fr, events=['start', 'end']):
            tag = strip_tag(elem.tag)

            if event == 'start':
                # start new row for new page
                if tag == 'page':
                    row = dict()
            else:
                # on event end collect data and meta data
                if tag == 'title':
                    row[TITLE], row[DESCRIPTION] = split_title(elem.text)
                elif tag == 'ns':
                    # namespace 0 == article
                    if int(elem.text) == 0:
                        is_article = True
                        row[DATASET] = CORPUS
                elif tag == 'id':
                    # only accept the first ID
                    if ID not in row:
                        row[ID] = int(elem.text)
                        row[ID2] = 0
                elif tag == 'timestamp':
                    row[TIME] = datetime.strptime(
                        elem.text.replace('Z', 'UTC'), '%Y-%m-%dT%H:%M:%S%Z'
                    )
                    # 2018-07-29T18:22:20Z
                elif tag == 'redirect':
                    is_redirect = True
                    row[SUBSET] = "REDIRECT"
                    row[LINKS], row[DESCRIPTION] = split_title(elem.get('title'))
                    row[TAGS] = (row[LINKS], row[DESCRIPTION])
                elif tag == 'text':
                    # accept only if namespace == 0
                    if is_article:
                        if not is_redirect:
                            row[SUBSET] = "ARTICLE"
                            row[TEXT], row[LINKS], row[TAGS] = parse_markdown(elem.text.strip())
                        else:
                            row[TEXT] = elem.text.strip()
                # write and close row, reset flags etc. on closing page tag
                elif tag == 'page':
                    # handle only if namespace == 0
                    if is_article:
                        # calculate hash key
                        # take care: hash function in python ist non-deterministic !!!
                        row[HASH] = hash(tuple([row[key] for key in META]))
                        # increment article counter
                        article_count += 1
                        # append on list of rows
                        rows.append(row)
                        # write to csv or update dataframe every m articles
                        if (article_count % batch_size) == 0:
                            batch_writer.write_rows(rows)
                            # reset rows
                            rows = []
                        # print status every n articles
                        if (article_count % print_every) == 0:
                            print(locale.format("%d", article_count, grouping=True))

                    # reset page state
                    row = is_article = is_redirect = False

                elem.clear()

            if article_count == iterations:
                break

    batch_writer.write_rows(rows)


# --- Regex patterns for the following Markdown parser ---

# matches against:
# [[Kategorie:Soziologische Systemtheorie]], [[Kategorie:Fiktive Person|Smithee, Alan]]
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
re_meta = re.compile(
    r"(%s|%s|%s|%s|%s|%s)" % (chars, emph, bullet, bullet2, meta, footer), re.MULTILINE
)

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
    This is actucally not a real parser since it keeps no internal states. Therefore nested
    structures are a bit of a problem and a few artifacty may remain. Also the regexes are a bit
    nasty and need to read the text multiple times. Maybe I'm doing a new version at some point,
    but for the time being it's working sufficiently well.
    """
    # replace html escaping
    text = unescape(text)

    # extract categories
    categories = re_category.findall(text)

    # remove formatting tags and ref/math tags with content
    # This regex is actually working way better than the w3lib.remove_tags[_with_content]
    # implementations. # It's ~1.5x faster and keeps all wanted content, while the w3lib methods
    # introduce problems with some self-closing xml-tags. Of course lxml/beautifulsoup would be
    # another option.
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
        if len(split) > 1:
            token = split[1]
            link, category = split_title(split[0])
            link = None if token == link else link
        else:
            token = split[0]
            link = category = None
        links.append((token, link, category))
        return token

    text = re_link.sub(replace_links, text)

    # repeat for nested structures, performance-wise not perfect
    n = 1
    while n > 0:
        text, n = re_infobox.subn('', text)

    text = re_lf.sub('\n', text)
    return text.strip(' \n}{'), links, tuple(categories)


if __name__ == '__main__':
    t0 = time()
    print("Starting ...")
    scale = 1000
    parse_xml(IN_PATH, OUT_PATH + '_links_optimized',
              iterations=100,
              batch_size=100*scale,
              print_every=10*scale,
              )
    t1 = int(time() - t0)
    print("all done in", hms_string(t1))
