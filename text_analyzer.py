from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist
from nltk.corpus import stopwords

class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.tokenized_sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def extract_keywords(self, num_keywords=5):
        words = word_tokenize(self.text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        word_freq = FreqDist(filtered_words)
        keywords = word_freq.most_common(num_keywords)
        return keywords
