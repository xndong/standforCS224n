# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:23:07 2019

@author: DongXiaoning
"""


import re

# bigram
def generate_ngrams(corpus, n=2):
    # Convert to lowercases
    corpus = corpus.lower()
    
    # Replace all none alphanumeric characters with spaces
    corpus = re.sub(r'[^a-zA-Z0-9\s]', ' ', corpus)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in corpus.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    # Tokens[i:] is a list
    # There are n lists
    # These n lists comprise a list
    # We zip the list by elementwise symbol '*', our list's element are n lists.
    ngrams = zip(*[tokens[i:] for i in range(n)])  # zip自动截断长的sequence，until和短的sequence一样长
    token_npairs = [" ".join(ngram) for ngram in ngrams]
    return tokens,token_npairs

def word_count(tokens,word):
    word_count = 0
    for token in tokens:
        if token == word:
            word_count+=1
    return word_count

def pair_count(token_pairs,pair):
    pair_count =0
    for token_pair in token_pairs:
        if token_pair == pair:
            pair_count+=1
    return pair_count

def ngram_compute(tokens,token_npairs,sentence,n=2):
    sentence_probability = 1.0 
    words,pairs = generate_ngrams(sentence,2)
    for word,pair in zip(words,pairs):
        word_num = word_count(tokens,word)
        pair_num = pair_count(token_npairs,pair)
        if pair_num != 0:
            probability = pair_num / word_num
        else:
            probability = 0.01
        sentence_probability *= probability 
    return sentence_probability

def ngram_predict(tokens,token_npairs,sentence,vocabulary,n=2):
    word_probability_dic = {}
    sentence_probability = ngram_compute(tokens,token_npairs,sentence,2)
    sentence = [token for token in sentence.split(" ") if token != ""]
    for v in vocabulary:
        word_num = word_count(tokens,sentence[-1])
        pair_num = pair_count(token_npairs,sentence[-1] + " " + v)
        if pair_num != 0:
            probability = pair_num / word_num
        else:
            probability = 0.0001
        word_probability_dic[v] = sentence_probability * probability     
    return word_probability_dic

def main():
    corpus = "Natural-language processing (NLP) is af computer science " \
    "and artificial intelligence concerned with the interactions " \
    "between natural language computer and human (natural) languages. Naturale processing (NLP) is an area of computer science " \
    "and artificial intelligence concerned with the interactions " \
    "between compuman (natural) languages. Natural-language processing (NLP) isf computer science " \
    "and artificial intelligence concerned with the interactions " \
    "between computers and human (natural) languages. natural language natural language processing natural language natural language natural language Natural-language processing (NLP) is an area of computer science " \
    "and artificialigence concerned with the interactions " \
    "between computers and human (natural) languages.natural language processing natural language processing natural language processing natural language processing natural language processingnatural language processing natural language processing  Natural-language processing (NLP) is an area of computer science " \
    "and artificiigence concerned with the interactions " \
    "between computers and human (natural) languages."
    tokens,token_npairs = generate_ngrams(corpus,2)
    print(ngram_compute(tokens,token_npairs,"natural language processing is an",2))
    print(ngram_predict(tokens,token_npairs,"natural language",["processing","computer","science"],2))

if __name__ =="__main__":
    main()