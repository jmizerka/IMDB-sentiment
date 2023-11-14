import random
import re

import numpy as np

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


def tokenize(text):
    return word_tokenize(text)


def clean_data(text):
    return re.sub(r'<br />', '', text)


def flt_stop_words(text):
    stop_words_english = set(stopwords.words("english"))
    return [word for word in text if word not in stop_words_english]


def pos_tagging(text):
    tagged = pos_tag(text)
    tagged_flt = [word for word, tag in tagged if tag.startswith('JJ') or tag.startswith('RB')]
    return tagged_flt


def stem_text(text):
    ps = PorterStemmer()
    stems = [ps.stem(word) for word in text]
    return stems


def bag_of_words(reviews, num_of_words):
    words_combined = []
    for review in reviews:
        words_combined.extend(review)
    common_words = np.array(nltk.FreqDist(words_combined).most_common(num_of_words))[:, 0]
    return common_words


def find_features(review, common_words):
    words = set(review)
    features = {}
    for w in common_words:
        features[w] = (w in words)
    return features


def preprocess_data(data, num_of_features=500):
    data['review'] = data['review'].apply(clean_data)
    data['review'] = data['review'].apply(tokenize)
    data['review'] = data['review'].apply(flt_stop_words)
    data['review'] = data['review'].apply(pos_tagging)
    data['review'] = data['review'].apply(stem_text)
    most_common_words = bag_of_words(data['review'], num_of_features)
    feature_set = [(find_features(review[1][0], most_common_words), review[1][1]) for review in data.iterrows()]
    random.shuffle(feature_set)
    return data, most_common_words, feature_set
