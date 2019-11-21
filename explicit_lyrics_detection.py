import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# dictionary is derived from a training set of lyrics using the make_dictionary() method
from sklearn.metrics import precision_recall_fscore_support

SWEARWORDS_DICTIONARY = ['motherfuckers', 'bitches', 'niggas', 'hoes', 'nigga', 'motherfuckin', 'niggaz', 'motherfucker', 'glock', 'gat', 'pussy', 'thug', 'fuckin', 'hoe', 'bitch', 'blunt', 'gangsta', 'homie', 'pimp', 'dick', 'holla', 'clip', 'homies', 'haters', 'beef', 'rapper', 'weed', 'fo', 'cock', 'mac', 'rappers', 'fuck']

def make_dictionary(train_file, size=32, output_weights=False, ngram_range=(1,1), max_features=1000):
    explicit_tokens = []
    clean_tokens = []
    labels_train = []
    # assumes train_file as tsv format with column 0: labels (0, 1), column 1: text
    with open(train_file, 'r') as f:
        for line in f.readlines():
            tabs = line.split('\t')
            label = tabs[0]
            text = tabs[1]
            unique_tokens = list(set(text.split()))
            if label == '0':
                clean_tokens.extend(unique_tokens)
            else:
                explicit_tokens.extend(unique_tokens)
            labels_train.append(label)
    explicit_text = [' '.join(explicit_tokens)]
    clean_text = [' '.join(clean_tokens)]
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    exp_counts = vectorizer.fit_transform(explicit_text) / labels_train.count('1')
    clean_counts = vectorizer.transform(clean_text) / labels_train.count('0')
    exp_to_clean_ratio = exp_counts / clean_counts

    word_ratio = [(word, exp_to_clean_ratio[0, i]) for (i, word) in enumerate(vectorizer.get_feature_names()) if
                  exp_to_clean_ratio[0, i] < np.inf]
    word_ratio_dict = dict(word_ratio)
    dictionary = [w for (w, _) in sorted(word_ratio, reverse=True, key=lambda x: x[1])][:size]
    if output_weights:
        return dict([(w,word_ratio_dict[w]) for w in dictionary])
    return dictionary

def texts_labels(tsv_file):
    texts = []
    labels = []
    with open(tsv_file, 'r') as f:
        for line in f.readlines():
            tabs = line.split('\t')
            texts.append(tabs[1])
            labels.append(tabs[0])
    return texts, labels

def dictionary_lookup_classifier(dev_file, results_file, dictionary):
    texts_dev, labels_dev = texts_labels(dev_file)
    labels_pred = []
    dictionary_bow = set(dictionary)
    for text, label in zip(texts_dev, labels_dev):
        text_bow = set(text.split())
        prediction = '1' if text_bow.intersection(dictionary_bow) else '0'
        labels_pred.append(prediction)
    pd.to_pickle(labels_pred, results_file)
    return precision_recall_fscore_support(y_true=labels_dev, y_pred=labels_pred)

def predict_explicitness(text):
    dictionary_bow = set(SWEARWORDS_DICTIONARY)
    text_bow = set(text.split())
    prediction = 'explicit' if text_bow.intersection(dictionary_bow) else 'NOT explicit'
    return prediction