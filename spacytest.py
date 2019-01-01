import spacy

nlp = spacy.load('de')
from spacy.lang.de.examples import sentences

print(sentences)
docs = nlp('. '.join(sentences + ["auf'm drop'n Haupt-Haus"]))

for token in docs:
    print(token.i, token.text)

