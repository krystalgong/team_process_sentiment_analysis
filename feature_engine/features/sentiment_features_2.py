# imports

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import nltk
from nltk.corpus import stopwords

import numpy as np
from scipy.special import softmax

# Preprocess text (username and link placeholders)
stop_words = set(stopwords.words('english'))
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        # Remove stop words and the word 'asshole'
        t = t.lower()
        if t not in stop_words and 'asshole' not in t:
            new_text.append(t)
    return " ".join(new_text)

# load pre-trained models

# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
MODEL1 = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
config1 = AutoConfig.from_pretrained(MODEL1)
model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1)

# https://huggingface.co/philschmid/distilbert-base-multilingual-cased-sentiment-2
MODEL2 = f"philschmid/distilbert-base-multilingual-cased-sentiment-2"
tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
config2 = AutoConfig.from_pretrained(MODEL2)
model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2)

# TODO - DEFINE YOUR FEATURE EXTRACTOR HERE
def get_sentiment_1(text):
    # preprocess text
    text = preprocess(text)
    # tokenize text
    encoded_input = tokenizer1(text, return_tensors='pt')
    # get model output
    output = model1(**encoded_input)
    # get scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # get ranking
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    # get positive, negative and neutral scores
    for i in range(scores.shape[0]):
        # get label and corresponding score
        label = config1.id2label[ranking[i]]
        s = scores[ranking[i]]
        if label == 'positive':
            positive = s
        elif label == 'negative':
            negative = s
        elif label == 'neutral':
            neutral = s
    
    # return dictionary with positive, negative and neutral scores
    return ({'positive': positive, 'negative': negative, 'neutral': neutral})

def get_sentiment_2(text):
    # preprocess text
    text = preprocess(text)
    # tokenize text
    encoded_input = tokenizer2(text, return_tensors='pt')
    # get model output
    output = model2(**encoded_input)
    # get scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # get ranking
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    # get positive, negative and neutral scores
    for i in range(scores.shape[0]):
        # get label and corresponding score
        label = config2.id2label[ranking[i]]
        s = scores[ranking[i]]
        if label == 'positive':
            positive = s
        elif label == 'negative':
            negative = s
        elif label == 'neutral':
            neutral = s
    
    # return dictionary with positive, negative and neutral scores
    return ({'positive': positive, 'negative': negative, 'neutral': neutral})