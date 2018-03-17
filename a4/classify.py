"""
classify.py
"""

# Imports
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pickle
import re

def download_afinn_data():
    # Download the AFINN lexicon, unzip, and read the latest word list in AFINN-111.txt
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    return afinn


def afinn_sentiment2(terms, afinn, verbose=False):
    
    pos = 0
    neg = 0
    
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg
    
def tokenize(text):
    
    return re.sub('\W+', ' ', text.lower()).split()

def pos_neg(tweets,tokens,afinn):
    
    positives = []
    negatives = []
    mixed   = []
    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sentiment2(token_list, afinn)
        if pos > neg:
            positives.append((tweet['text'], pos, neg))
        if neg > pos:
            negatives.append((tweet['text'], pos, neg))
        elif neg==pos:
            mixed.append((tweet['text'], pos, neg))
    return positives, negatives, mixed



def main():
    
    with open('tweets.pickle', 'rb') as f:
        tweets = pickle.load(f)
    
    afinn = download_afinn_data()
    
    tokens = [tokenize(t['text']) for t in tweets]

    positives, negatives, mixed =pos_neg(tweets,tokens,afinn)

    with open('classify.pickle', 'wb') as f:
        pickle.dump((positives,negatives,mixed), f, pickle.HIGHEST_PROTOCOL)
    
    
    
if __name__ == '__main__':
    main()