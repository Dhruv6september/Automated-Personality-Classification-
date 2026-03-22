import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.data.path.append("C:/Users/Mickey/nltk_data")

stop_words = set(stopwords.words('english'))

def preprocess(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("|||", " ")

    #Lowercase
    text = text.lower()

    #Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    #Tokenize
    words = word_tokenize(text)

    #Keep important words
    words = [w for w in words if w not in stop_words or w in ["not", "no"]]

    return " ".join(words)