# -*- coding: utf-8 -*-
'''
Filename : LoadData.py
Function : Preprocess source data and transform to standard fotmat
           See GenerateData.py for the steps of preprocess
'''

import re
import os
import json
import codecs
import numpy as np
import pandas as pd
from collections import Counter
from code.process.data import Instance,Triple 
import string

''' Load Entity2id & Relation2id file '''
def loadTerm2id (dataPath):
    termDict = {}
    with open (dataPath,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            termName = line[0]
            termNum  = line[1]
            termDict[termName] = termNum
    return termDict

''' clean the word '''
def normWord (word):
    word = word.strip().lower()
    word = re.sub(u'\s+', '', word, flags=re.U)  # match all the blank character
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)

    if word.isdigit():
        word = '1'
    else:
        temp = word
        for char in word:
            if char not in string.printable:
                temp = temp.replace(char, '*')
        word = temp
    return word


''' Translate sentence into ID in wordDict'''
def sentence2id (sentence,wordDict,headStart,headEnd,tailStart,tailEnd):
    sentenceID = []
    headID = []
    tailID = []
        
    # replace entity with a special name
    head = sentence[headStart:headEnd]
    tail = sentence[tailStart:tailEnd]
    if headStart< tailStart:
        sentence = sentence[:headStart] + ' HEADENTITY ' + sentence[headEnd:tailStart] + ' TAILENTITY ' + sentence[tailEnd:]
    else:
        sentence = sentence[:tailStart] + ' TAILENTITY ' + sentence[tailEnd:headStart] + ' HEADENTITY ' + sentence[headEnd:]
    
    # normalize sentence
    for c in string.punctuation:
        sentence = sentence.replace(c,' '+c+' ')
        head = head.replace(c,' '+c+' ')
        tail = tail.replace(c,' '+c+' ')
    sentence = sentence.split()
    head = head.split()
    tail = tail.split()
    
    # normalize entity
    for word in head:
        word = normWord(word)
        if word not in wordDict:
            wordDict[word] = len(wordDict) + 1  #the ID zero should be retained for padding words
        headID.append(wordDict[word])
    for word in tail:
        word = normWord(word)
        if word not in wordDict:
            wordDict[word] = len(wordDict) + 1  
        tailID.append(wordDict[word])
    
    # calculate the position of head/tail entity in the sentence
    # translate sentence with words into sentence with ids
    for i,word in enumerate(sentence):
        if word == 'HEADENTITY':
            newHeadStart = i
            newHeadEnd = i + len(headID)
            sentenceID.extend(headID)
            continue
        if word == 'TAILENTITY':
            newTailStart = i
            newTailEnd = i + len(tailID)
            sentenceID.extend(tailID)
            continue
        word = normWord(word)
        if word not in wordDict:
            wordDict[word] = len(wordDict) + 1 
        sentenceID.append(wordDict[word])
    return sentenceID,newHeadStart,newHeadEnd,newTailStart,newTailEnd,wordDict

'''
Load train.txt File into a list
train.txt format:
head \t tail \t relation \t sentenceNums
offset1_start1 \t offset1_end1 \t offset1_start2 \t offset1_end2 \t sentence1
offset2_start1 \t offset2_end1 \t offset2_start2 \t offset2_end2 \t sentence2
...
head \t tail \t relation \t sentenceNums
offset1_start1 \t offset1_end1 \t offset1_start2 \t offset1_end2 \t sentence1
offset2_start1 \t offset2_end1 \t offset2_start2 \t offset2_end2 \t sentence2
offset3_start1 \t offset3_end1 \t offset3_start2 \t offset3_end2 \t sentence3
...

'''
def loadTrainTriples (dataPath,entityDict,relationDict,wordDict):
    triples = []   # store all the triples with texts
    with open (dataPath,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            head = int(entityDict[line[0]])
            tail = int(entityDict[line[1]])
            relation = int(relationDict[line[2]])
            sentenceNum = int(line[3])
            triple = Triple (
                head = head,
                tail = tail,
                relation = relation,
                instance = []
            )
            for _ in range (sentenceNum):
                sentenceInform = f.readline().strip().split('\t')
                offsetStartHead   = int(sentenceInform[0])
                offsetEndHead     = int(sentenceInform[1])
                offsetStartTail   = int(sentenceInform[2])
                offsetEndTail     = int(sentenceInform[3])
                sentence          = sentenceInform[4].strip() #.split(' ')
                sentence,headStart,headEnd,tailStart,tailEnd,wordDict = sentence2id(sentence = sentence,
                                                                                    headStart = offsetStartHead,
                                                                                    headEnd = offsetEndHead,
                                                                                    tailStart = offsetStartTail,
                                                                                    tailEnd = offsetEndTail,
                                                                                    wordDict = wordDict)
                instance = Instance (
                    offsetStartHead = headStart,
                    offsetEndHead = headEnd,
                    offsetStartTail = tailStart,
                    offsetEndTail = tailEnd,
                    sentence = sentence
                )
                triple.instance.append(instance)
            triples.append(triple)
            
    return triples,wordDict

''' 
Generate the embedding matrix of word embedding
If the pretrain embeding is not given, the word embedding is randomly initialized from -0.5 to 0.5
'''
def loadPretrainW2V (embedPath,wordDict,embeddingDim):
    wordNum = len(wordDict) + 1    #plus one for 0(padding word)
    wordEmbedding = np.random.uniform(-0.5,0.5,size = (wordNum,embeddingDim))
    wordEmbedding[0] = np.zeros(shape = embeddingDim ,dtype = 'float32')
    word2vec = {}
    if os.path.exists(embedPath):
        with open (embedPath,'r') as f:
            for line in f:
                line = line.strip().split()
                word = line[0].lower()     #uncased
                embed = np.asarray(line[1:],dtype = 'float32')
                word2vec[word] = embed
        for word in wordDict:
            if word in word2vec:
                wordEmbedding[wordDict[word]] = word2vec[word]
    return wordEmbedding

''' 
Generate the embedding matrix of entity & relation embedding
If the pretrain embeding is not given, the word embedding is randomly initialized from -0.5 to 0.5
'''
def loadPretrainE2V (embedPath,entityDict,embeddingDim):
    entityNum = len(entityDict)
    entityEmbedding = np.random.uniform(-0.5,0.5,size = (entityNum,embeddingDim))
    entityEmbedding[0] = np.zeros(shape = embeddingDim ,dtype = 'float32')
    if os.path.exists(embedPath):
        with open (embedPath,'r') as f:
            for line in f:
                line = line.strip().split()
                entityID = int (line[0])     #format:  entityID \t 0.1 \t 0.2 \t 0.15 ...
                embed = np.asarray(line[1:],dtype = 'float32')
                entityEmbedding[entityID] = embed
    return entityEmbedding

''' Split all data into train & evaluation data'''
def splitData (allTriples,splitRate):
    tripleNum = len(allTriples)
    np.random.shuffle(allTriples)
    splitTrainNum = int(splitRate*tripleNum)

    return allTriples[splitTrainNum:],allTriples[:splitTrainNum]

    
