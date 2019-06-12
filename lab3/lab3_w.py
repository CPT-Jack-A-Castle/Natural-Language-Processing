#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:01:22 2019

@author: walt
"""

# For imports
import itertools
from sklearn.metrics import f1_score
from collections import Counter
import random


# Dictionaries
cw_cl_counts = {}
pl_cl_counts = {}

random.seed(448)

def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

trainCorpus = load_dataset_sents("train.txt")
testCorpus = load_dataset_sents("test.txt")

# Current word-current label (1)
def phi_1_courps(inputCorpus):
    rawValues = []
    for sentence in inputCorpus:
        for word in sentence:
            
            wordBond = (word[0] + "_" +  word[1])
            rawValues.append(wordBond)
    
    uniqueValues = Counter(rawValues)
    
    for k,v in uniqueValues.items():
        if v >= 3:
            cw_cl_counts.update({k : v})
            
    return cw_cl_counts

def phi_1(x, y, cw_cl_counts):
    full = []
    phi_1_dic = {}
    for i in range(len(x)):
        wordBond = (x[i] + "_" +  y[i])
        full.append(wordBond)
    uniqueValues = Counter(full)
    for k,v in uniqueValues.items():
        if k in cw_cl_counts.keys():
            phi_1_dic.update({k : v})
        else:
            v = 0 
            phi_1_dic.update({k : v})
    return phi_1_dic


# Current word-current label (2)
def phi_2_courps(inputCorpus):
    rawValues = []
    for sentence in inputCorpus:
        rawTags = []
        for word in sentence:
            tag = (word[1])
            rawTags.append(tag)
        for i in range(len(rawTags)):
            if i <= 0:
                tagBond = ("None" + "_" + rawTags[i])
                rawValues.append(tagBond)
            else:
                tagBond = (rawTags[i-1] + "_" + rawTags[i])
                rawValues.append(tagBond)
    
    uniqueValues = Counter(rawValues)
    
    for k,v in uniqueValues.items():
        if v >= 3:
            pl_cl_counts.update({k : v})
            
    return pl_cl_counts

def phi_2(x, y, pl_cl_counts):
    rawTags = []
    fullValues = []
    phi_2_dic = {}
    for i in range(len(x)):
        tag = (y[i])
        rawTags.append(tag)
    for i in range(len(rawTags)):
        if i <= 0:
            tagBond = ("None" + "_" + rawTags[i])
            fullValues.append(tagBond)
        else:
            tagBond = (rawTags[i-1] + "_" + rawTags[i])
            fullValues.append(tagBond)
            
    uniqueValues = Counter(fullValues)
    for k,v in uniqueValues.items():
        if k in pl_cl_counts.keys():
            phi_2_dic.update({k : v})
        else:
            v = 0 
            phi_2_dic.update({k : v})
    return phi_2_dic

def train(trainCorpus, phiType):
    print("Starting training", phiType, "now....")
    w_1 = {}
    w_2 = {}
    iteration = 1
    while iteration != 6:
        random.shuffle(trainCorpus)
        for sentence in trainCorpus:
            xList = []
            yList = []
            for word in sentence:
                xList.append(word[0])
                yList.append(word[1])
            
            possibleTags = list(itertools.product(['ORG', 'MISC', 'PER', 'LOC', 'O'], repeat=len(xList)))
            
            highestValue_phi_1 = 0
            highestValue_phi_2 = 0
            bestPair_phi_1 = []
            bestPair_phi_2 = []
            
            for tag in possibleTags:
                wSum_phi_1 = 0
                wSum_phi_2 = 0
                combinded = {}
                
                phi_sum_1 = phi_1(xList, tag, cw_cl_counts)
                phi_sum_2 = phi_2(yList, tag, pl_cl_counts)
                
                combinded = phi_sum_1.copy()
                combinded.update(phi_sum_2)

# Phi_1 ----------------------------------------------------------------                
                for k,v in phi_sum_1.items():
                    if k in w_1:
                        wSum_phi_1 += w_1[k] * v
                    else:
                        wSum_phi_1 += 0
                
                if wSum_phi_1 >= highestValue_phi_1:
                    highestValue_phi_1 = wSum_phi_1
                    bestPair_phi_1 = tag

# Phi_2 ----------------------------------------------------------------
                for k,v in combinded.items():
                    if k in w_2:
                        wSum_phi_2 += w_2[k] * v
                    else:
                        wSum_phi_2 += 0
                
                if wSum_phi_2 >= highestValue_phi_2:
                    highestValue_phi_2 = wSum_phi_2
                    bestPair_phi_2 = tag                    
                    
# Phi_1 ----------------------------------------------------------------
                    
            if bestPair_phi_1 != yList:
                phi_pre_1 = phi_1(xList, bestPair_phi_1, cw_cl_counts)
                phi_true_1 = phi_1(xList, yList, cw_cl_counts)
                
                for k,v in phi_pre_1.items():
                    if k not in w_1:
                        w_1.update({k : 0})
                        
                    w_1[k] = w_1[k] - v
                        
                for k,v in phi_true_1.items():
                    if k not in w_1:
                        w_1.update({k : 0})
                        
                    w_1[k] = w_1[k] + v

# Phi_2 ----------------------------------------------------------------

            if bestPair_phi_2 != yList:
                full_pre = {}
                full_true = {}
                
                phi_pre_1 = phi_1(xList, bestPair_phi_1, cw_cl_counts)
                phi_true_1 = phi_1(xList, yList, cw_cl_counts)
                
                phi_pre_2 = phi_2(xList, bestPair_phi_2, pl_cl_counts)
                phi_true_2 = phi_2(xList, yList, pl_cl_counts)
                
                full_pre = phi_pre_1.copy()
                full_true = phi_true_1.copy()
                
                
                full_pre.update(phi_pre_2)
                full_true.update(phi_true_2)
                
                for k,v in full_pre.items():
                    if k not in w_2:
                        w_2.update({k : 0})
                        
                    w_2[k] = w_2[k] - v
                        
                for k,v in full_true.items():
                    if k not in w_2:
                        w_2.update({k : 0})
                        
                    w_2[k] = w_2[k] + v
                    
        iteration += 1
        
        if phiType == 1:
            trainingWeight = w_1
        elif phiType == 2:
            trainingWeight = w_2

    return trainingWeight



def predict(List, w, phiType):
    
    possibleTags = list(itertools.product(['ORG', 'MISC', 'PER', 'LOC', 'O'], repeat=len(List)))
    bestPair = []
    highestValue = 0
    
    for tag in possibleTags:
        wSum = 0
        
# Phi_1 ---------------------------------------------------------------- 
        if phiType == 1:
            phi_sum_1 = phi_1(List, tag, cw_cl_counts)
            for k,v in phi_sum_1.items():
                if k in w:
                    wSum += w[k] * v
                else:
                    wSum += 0
            
            if wSum >= highestValue:
                highestValue = wSum
                bestPair = tag

# Phi_2 ---------------------------------------------------------------- 
        elif phiType == 2:
            phi_sum_1 = phi_1(List, tag, cw_cl_counts)
            phi_sum_2 = phi_2(List, tag, pl_cl_counts)
            
            phi_sum_1.update(phi_sum_2)
            
            for k,v in phi_sum_1.items():
                if k in w:
                    wSum += w[k] * v
                else:
                    wSum += 0
            
            if wSum >= highestValue:
                highestValue = wSum
                bestPair = tag
    
    return bestPair


def test(testCorpus):
    print("Starting test now....")
    correctY_phi_1 = [] 
    predictedY_phi_1 = []
    
    correctY_phi_2 = [] 
    predictedY_phi_2 = []
    
    correct_phi_2 = []
    yList = []
    
    for sentence in testCorpus:
        xList = []
        
        for word in sentence:
            xList.append(word[0])
            yList.append(word[1])

            correctY_phi_1.append(word[1])
            correctY_phi_2.append(word[1])
        
        predictedY_phi_1 += predict(xList, wight_phi_1, 1)
        predictedY_phi_2 += predict(xList, wight_phi_2, 2)  
    
    full_prediction = predictedY_phi_1 + predictedY_phi_2
    correct_phi_2 = correctY_phi_1 + correctY_phi_2
    
    print(len(correct_phi_2))
    print(len(full_prediction))
    
    f1_micro = f1_score(correctY_phi_1, predictedY_phi_1, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC', 'O'])
    print(f1_micro)
    
    f2_micro = f1_score(correctY_phi_2, predictedY_phi_2, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC', 'O'])
    print(f2_micro)

cw_cl_counts = phi_1_courps(trainCorpus)
pl_cl_counts = phi_2_courps(trainCorpus)

wight_phi_1 = train(trainCorpus, 1)
wight_phi_2 = train(trainCorpus, 2)
test(testCorpus)



