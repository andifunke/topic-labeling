import re

from spacy.symbols import IDS

# from iwnlp.iwnlp_wrapper import IWNLPWrapper

# LEMMATIZER = IWNLPWrapper(
#     lemmatizer_path=os.environ['VIRTUAL_ENV']+'/data/IWNLP.Lemmatizer_20170501.json')

STREET_NAME_LIST = ['strasse', 'straße', 'str', 'str.', 'platz', 'gasse', 'allee', 'ufer', 'weg']
STREET_PATTERN = re.compile(r"str(\.|a(ss|ß)e)?$", re.IGNORECASE)


def _check_pos(pos, token_pos):
    """Check if token_pos id in pos. If there is no pos return True.
    
    Arguments:
        pos {iterable} -- Iterable object of strings. POS tags to filter. If empty return true.
        token_pos {string} -- SpaCy POS
    
    Returns:
        boolean -- False if token_pos not in POS an POS not empty
    """
    return (not pos) or (token_pos in pos)

def unzip_dicts(key, list_of_dicts):
    return [obj[key] for obj in list_of_dicts]

def handle_capitalize(string):
    if string.isupper():
        return string
    else:
        return string.capitalize()

def parse_street_names(token, street_pattern):
    """Parse a given spaCy token to get street names right. If no street in it or nor IDS['LOC'] the
    spaCy lemma is used.
    
    Arguments:
        token {spaCy Token}
        street_pattern {regex pattern} -- Matches will be replaced with 'straße'
    
    Returns:
        String -- The parsed lemma for a given token.
    """

    lemma,n = re.subn(street_pattern, 'straße', token.text)
    if token.ent_type == IDS['LOC'] or n > 0:
        if lemma.count(' ') == 1 and lemma.count('-') == 0:
                lemma = lemma.replace(' ', '').capitalize()
        elif lemma.count(' ') >= 1 or lemma.count('-') >= 1:
            lemma = lemma.replace(' ','-')
            lemma = map(handle_capitalize, re.split(r'(\w+)', lemma))
            lemma = ''.join(lemma)
        else:
            lemma = token.lemma_
    else:
        lemma = token.lemma_
    return lemma

def get_lemma(token, street_pattern, iwlnp_lemma=False, street_parse=True):
    """Get the lemma of given word. Possible to parse street names.
        If IWNLP doesn't provide a lemma spaCy is used.
    
    Arguments:
        token {string} -- token to parse
        street_pattern {string} -- regex pattern to delete from token
    
    Keyword Arguments:
        iwlnp_lemma {boolean} -- Specify whether to use IWNLP lemmas first (default: {True})
        street_parse {boolean} -- Specify whether to parse street names (default: {False})
    
    Returns:
        string -- lemma of given token
    """
    lemma = None
    if iwlnp_lemma:
        lemma = LEMMATIZER.lemmatize(token.text,pos_universal_google=token.pos_)
    if not lemma:
        if street_parse:
            return parse_street_names(token, street_pattern)
        return token.lemma_
    else:
        return lemma[0]
    
def change_lemmas(doc, iwlnp_lemma=False, street_parse=True):
    """Change the lemmas of tokens in given doc.
    
    Arguments:
        doc {spacy doc} -- Spacy document
    
    Keyword Arguments:
        iwlnp_lemma {boolean} -- Specify whether to use IWNLP lemmas first (default: {True})
        street_parse {boolean} -- Specify whether to parse street names (default: {False})
    
    Returns:
        spacy doc
    """
    for token in doc:
        token.lemma_ = get_lemma(token, STREET_PATTERN, iwlnp_lemma, street_parse)
    return doc

def tokens_to_dict(doc, iwlnp_lemma=False, street_parse=True):
    """Structure the given list of spaCy tokens to a dictionary for JSON.
    
    Arguments:
        doc {spacy doc} -- Spacy document
    
    Keyword Arguments:
        iwlnp_lemma {boolean} -- Specify whether to use IWNLP lemmas first (default: {True})
        street_parse {boolean} -- Specify whether to parse street names (default: {False})
    
    Returns:
        list -- List of dictionaries ready to write to a JSON file.
    """

    json_tokens = []
    for token in doc:
        json_tokens.append({'Text': token.text,
                            'Lemma': get_lemma(token, STREET_PATTERN, iwlnp_lemma, street_parse),
                            'POS': token.pos_,
                            'NER_type': token.ent_type_,
                            'NER_IOB': token.ent_iob_,
                            'Stop': token.is_stop
                            })
    return json_tokens
        
def parse_dependencies(doc, target_text_list):
    """Merge tokens of named entities which contains one of given target strings to one token.
    
    Arguments:
        doc {spaCy doc} -- A single document
        target_text_list {list} -- list of all lowercase target strings. E.g. ['straße', 'str']
    Returns:
        [type] -- [description]
    """

    for span in doc.ents:
        if span[-1].lower_ in target_text_list:
            span.merge(ent_type=IDS['LOC'], pos=IDS['PROPN'])
    return doc

def process_docs(nlp, documents, iwnlp_lemma=False, street_parse=True):
    """Process the content of the given list of suggestions and its comments with NLP.
    
    Arguments:
        nlp {SpaCy nlp} -- nlp object to process text with
        documents {list} -- list of dictionaries obtained from a JSON with suggestions of a online participation procedure.
    
    Keyword Arguments:
        iwnlp_lemma {boolean} -- Specify whether to use IWNLP lemmas first (default: {True})
        street_parse {boolean} -- Specify whether to parse street names (default: {False})
    
    Returns:
        list -- list of tokens compiled by spaCy
    """

    tokenized_list = []
    for doc in nlp.pipe(documents, batch_size=50):
        if street_parse:
            doc = parse_dependencies(doc, STREET_NAME_LIST)
        tokenized_list.append({'Tokens': tokens_to_dict(doc, iwnlp_lemma, street_parse), 'Text': doc.text})
    return tokenized_list

def process_blacklist(nlp, blacklist, iwnlp_lemma=False, street_parse=True):
    tokenized_list = []
    for doc in nlp.pipe(blacklist, batch_size=50):
        tokenized_list += [get_lemma(token, STREET_PATTERN) for token in doc]
    return {token.lower() for token in tokenized_list}

def filter_tokens(tokens, pos=None, lowercase=True, use_lemmas=True, filter_stopwords=False, blacklist=None):
    """Filters tokens from given documents.
    
    Arguments:
        documents {list of dicts} -- List of documents as dictionaries, each containing list of tokens.

    Keyword Arguments:
        pos {iterable} -- Specify which POS to filter. If None get all POS (default: {None})
        lowercase {boolean} -- Specify whether to convert tokens to lowercase (default: {True})
        use_lemmas {boolean} -- Specify whether to use lemmas instead of the raw token (default: {True})
        blacklist {iterable} -- Iterable object of words to blacklist i.e. to ignore (default: {None})
    
    Returns:
        list -- List with one list per document each containing filtered and processed tokens.
    """
    new_document = []
    for token in tokens:
        if _check_pos(pos,token['POS']):
            if use_lemmas:
                if token['Lemma']:
                    value = token['Lemma']
                else:
                    value = token['Text']
            else:
                value = token['Text']
            if lowercase:
                value = value.lower()
            if blacklist:
                if value in blacklist:
                    continue
            if (filter_stopwords and token['Stop']):
                continue
            new_document.append(value)
    return new_document
