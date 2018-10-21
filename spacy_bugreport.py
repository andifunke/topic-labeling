import spacy
nlp = spacy.load('en')
text = "This is the first sentence. This is the second sentence."
doc = nlp(text)
for token in doc:
    print(token.i, token.is_sent_start, token.text)
