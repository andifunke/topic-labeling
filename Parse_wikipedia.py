
# coding: utf-8

# ### Parse Wikipedia

# In[1]:


from time import time
from os.path import join
import csv
import locale
locale.setlocale(locale.LC_ALL, '')

from constants import DATA_BASE

# Warning: The xml.etree.ElementTree module is not secure against maliciously constructed data. 
# If you need to parse untrusted or unauthenticated data see XML vulnerabilities 
# (https://docs.python.org/3/library/xml.html#xml-vulnerabilities)
import xml.etree.ElementTree as ET

#import re
#from w3lib.html import remove_tags, remove_tags_with_content
from html import unescape


# In[2]:


PATH_WIKI_XML = join(DATA_BASE, 'dewiki')
FILENAME_WIKI = 'dewiki-latest-pages-articles.xml'
FILENAME_ARTICLES = 'articles.csv'

pathWikiXML = join(PATH_WIKI_XML, FILENAME_WIKI)
pathArticles = join(PATH_WIKI_XML, FILENAME_ARTICLES)


# In[3]:


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def strip_tag(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

def split_title(title):
    split = title.find('(', 1)
    split = None if split < 1 else split
    return title[:split], (title[split:] if split else '')


# In[4]:


def parse_xml(infile, outfile, iterations, print_every=1000):

    with open(infile, 'r') as fr, open(outfile, 'w') as fw:

        fields = ['id', 'timestamp', 'title', 'subtitle', 'text', 'categories', 'links', 'redirect']
        writer = csv.DictWriter(fw, fields, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        articleCount = 0
        row = None
        is_article = is_liste = is_redirect = False

        for event, elem in ET.iterparse(fr, events=['start', 'end']):
            tag = strip_tag(elem.tag)

            if event == 'start':
                # start new row
                if tag == 'page':
                    row = dict()
            else:
                if tag == 'title':
                    if elem.text.startswith('Liste von'):
                        is_liste = True
                    else:
                        row[tag], row['subtitle'] = split_title(elem.text)
                        #print(row[tag], end=': ')
                elif tag == 'ns' and int(elem.text) == 0:
                    if is_liste:
                        is_article = False
                    else:
                        is_article = True
                elif tag == 'id' and tag not in row:
                    row[tag] = elem.text
                elif tag == 'timestamp':
                    row[tag] = elem.text
                elif tag == 'redirect':
                    is_redirect = True
                    row[tag] = elem.get('title')
                elif tag == 'text' and is_article and not is_redirect:
                    row[tag], row['categories'], row['links'] = parse_text(elem.text)
                    if not row[tag]:
                        is_article = False
                # write and close row, reset flags
                elif tag == 'page':
                    if is_article:
                        writer.writerow(row)
                        articleCount += 1
                        # print status
                        if articleCount > 1 and (articleCount % print_every) == 0:
                            print(locale.format("%d", articleCount, grouping=True))
                    # reset everything
                    row = None
                    #print(is_article, is_liste, is_redicret)
                    is_article = is_liste = is_redirect = False
                elem.clear()

            if articleCount == iterations:
                break


# In[7]:


import re

# We will keep the header decorations to subdivide an article at a later stage

# matches against: [[Kategorie:Soziologische Systemtheorie]], [[Kategorie:Fiktive Person|Smithee, Alan]]
category = r"\[\[Kategorie:(?P<cat>[\w ]+)(?:\|.*)?\]\]"
re_category = re.compile(category)

# remove tags
refs =     r"<\s*(ref|math)[^>.]*?(?:\/\s*>|>.*?<\s*\/\s*(ref|math)\s*>)"
tags =     r"<(.|\n)*?>"
re_tags = re.compile(r"(%s|%s)" % (refs, tags), re.MULTILINE)

table =    r"{\|(?s:.*?)\|}"
re_table = re.compile(table)

# remove meta data: [[Datei:...]]
chars =    r"\xa0"
emph =     r"\'{2,}"
bullet =   r"^[\*:] *"
bullet2 =  r"^\|.*"
meta =     r"\[\[\w+:.*?\]\]"
footer =   r"== (Bibliographie|Literatur|Weblinks|Einzelnachweise) ==(?s:.)*"
re_meta = re.compile(r"(%s|%s|%s|%s|%s|%s)" % (chars, emph, bullet, bullet2, meta, footer), re.MULTILINE)

# => merge ^
remove = r'(' + r'|'.join([refs, tags, table, chars, emph, bullet, bullet2, meta, footer]) + r')'
re_remove = re.compile(remove, re.MULTILINE)

# matches against: [[Aristoteles]], [[Reductio ad absurdum|indirekten Beweis]]
wikilink = r"\[\[(.*?)\]\]"
re_link = re.compile(wikilink)

zitat =    r"{{Zitat(?:\||-.*?\|Übersetzung=)(?P<token>.*?)(?:\|.*?)?}}"
re_zitat = re.compile(zitat)

replace = r'(' + r'|'.join([wikilink, zitat]) + r')'
re_replace = re.compile(replace)

infobox =  r"{{.*?(?:}}|(?={{))"
re_infobox = re.compile(r"(%s)" % (infobox), re.DOTALL)

lf      = r"\n\n*\n(?!==)"
re_lf = re.compile(lf)


def parse_text(text):
    # replace html escapings
    text = unescape(text)

    # extract categories
    categories = re_category.findall(text)

    #text = re_remove.sub('', text)

    # remove formatting tags and ref/math tags with content
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
        #if 'token' in matchobj.groupdict():
        #    return "„" + matchobj.group('token') + "”"
        split = matchobj.group(1).split('|', 1) 
        links.append(split[0])
        return split[1] if len(split) > 1 else split[0]
    text = re_link.sub(replace_links, text)
    #text = re_replace.sub(replace_links, text)
    
    # repeat for nested structures, performancewise not perfect
    n = 1
    while n > 0:
        text, n = re_infobox.subn('', text)
    
    text = re_lf.sub('\n', text)
    return text.strip(' \n}{'), categories, links

get_ipython().run_line_magic('time', "parse_xml(pathWikiXML, pathArticles+'re2', 10000)")


# In[ ]:


get_ipython().system('jt -r')

