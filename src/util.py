from nltk import word_tokenize, WordNetLemmatizer
from textblob import TextBlob
import re

# From here:  https://stackoverflow.com/questions/47423854/sklearn-adding-lemmatizer-to-countvectorizer
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        regex_num_ponctuation = '(\d+)|([^\w\s])'
        # Ignore ine and two letter words
        regex_little_words = r'(\b\w{1,2}\b)'
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) 
                if not re.search(regex_num_ponctuation, t) and not re.search(regex_little_words, t)]


class TextBlobTokenizer:
    def __init__(self):
        pass
    def __call__(self, text):
        # Use textblob to extract some noun phrases
        phrases = list(TextBlob(text).noun_phrases)
        
        
        ret = []
        
        for phrase in phrases:
            # Fix spaces between words and apostrophes
            pat = r"\s*'\s*"
            phrase = re.sub(pat, '\'', phrase)
            phrase = phrase.strip()

            # Remove apostrophes at the start of a phrase
            pat = r"^'\S*"
            phrase = re.sub(pat, '', phrase)
            phrase = phrase.strip()
            

            ret.append(phrase)

        return ret


class Preprocessor:
    def __init__(self):
        pass
    def __call__(self, text):
        # remove all non alpha-numeric (with dashes) characters
        pat = r"[^a-zA-Z0-9-.!,?'\s]"
        text = re.sub(pat, '', text)
        return text