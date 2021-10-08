#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:06:24 2021

@author: kshama.singh
"""

import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = open("../WordToVector/wordToVector.txt", 'r').read()
#print(paragraph)


# Preprocessing the data
def processDataInText(paragraph):
    text = re.sub(r'\[[0-9]*\]',' ',paragraph)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text


# Preparing the sentences
def processData_SentencesTokenize(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


# Preparing the dataset from tokenized sentences
def processData_fromEachSentences_stopWords(sentences):
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    for i in range(len(sentences)):
        sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    return sentences


# Training the Word2Vec model
def processDataInModel(sentences):
    model = Word2Vec(sentences, min_count=1)
    return model


    
# Finding Word Vectors    
def getWordVectors(model, word):
    vector = model.wv[word]
    return vector


# Most similar words
def getSimilerWord(model, word):
    similar = model.wv.most_similar(word)
    return similar


Text = processDataInText(paragraph);
sentences = processData_SentencesTokenize(Text);
DataSet = processData_fromEachSentences_stopWords(sentences);
model = processDataInModel(DataSet);
vector = getWordVectors(model, 'extraordinary');
similar = getSimilerWord(model, 'extraordinary');


