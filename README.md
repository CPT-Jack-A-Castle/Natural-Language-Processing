# Natural-Language-Processing
NLP Course Work 

## Lab 1- Text Classification with the perceptron
Training a perceptron to perform a binary classification task

## Lab 2 - Language Modelling
Text completion using different n-gram language models
 - Unigram
 - Bigram
 - Bigram with add-1 smoothing
 
 Statistical method
 
## Lab 3 - Named Entity Recognition with the Structured Perceptron
Training a structured perceptron(Hidden Markov Models for structure) to perform NER task.
 - Features:
   - current_word-current_label
   - current_word-current_label + previous_label-current-label

Evaluation done using F1-Score from scikit-learn library

## Lab 4 - Viterbi and Beam Search
Improving of Structured Perceptron algorithm by using Viterbi to speed it up by over 2000% for exact search.

Better speeds achieved using the inexact method of Beam search but accuracy remained the same however.


## Lab 5 - Neural Language Modeling
Creating Language Mode.ls using Neural Methods. Using PyTorch to create word embeddings and using it to perform text completion.

With this method, we avoid the problem with zero probabilities and avoid the use of smoothing in our model.
