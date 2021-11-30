# -*- coding: utf-8 -*-
'''
Filename : dataloader.py
Function : pack the data into dataloader for batch learning
'''
import time
import sys
import math
import json
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import *
from code.utils.TensorDevice import LongTensorDevice,FloatTensorDevice,VariableDevice,ParameterDevice
from tqdm import tqdm
import copy

class tripleDataset(Dataset):
    def __init__(self, dataPath):
        super(Dataset, self).__init__()
        # Load entity-index dict and relation-index dict
        allTriples = pickle.load(open(dataPath,'rb'),encoding = 'iso-8859-1')
        self.triples = allTriples

        # Generate the positive instance
        self.positiveSamples = [[triple.head,triple.relation,triple.tail] for triple in self.triples]
    
    '''
    Name: paddingSentences
    Function: padding all the sentences to a fixed length
              padding all the triples to a fixed number of sentences
              generate the position embedding of head entity & tail entity in sentences
              generate the masking tensor of sentences & headEntities & tailEntities
    '''
    def paddingSentences(self,maxLen=0,maxSentenceNum=0):
        # if the maxLen or maxSentenceNum not given, calculate the max one
        tempMaxLen = 0
        tempMaxSentenceNum = 0
        self.sentenceSamples = []
        self.sentenceMaskings = []
        self.headEntityMaskings = []
        self.tailEntityMaskings = []
        print ('INFO: Calculate the max sentence number and the max sentence length.')
        for triple in tqdm(self.triples):
            if len(triple.instance) > tempMaxSentenceNum:
                tempMaxSentenceNum = len (triple.instance)
            for instance in triple.instance:
                if len(instance.sentence) > tempMaxLen:
                    tempMaxLen = len(instance.sentence)
        if not maxLen:
            maxLen = tempMaxLen
        if not maxSentenceNum:
            maxSentenceNum = tempMaxSentenceNum
        # padding sentence with zero
        print ('INFO: Generating and padding the sentences to the max sentence number and max sentence length.')
        for triple in tqdm(self.triples):
            newTriple = []
            sMaskings = []
            headEntityMasking = []
            tailEntityMasking = []
            for j,instance in enumerate(triple.instance):
                if maxLen>=len(instance.sentence):
                    newSentence = instance.sentence + [0]*(maxLen-len(instance.sentence))
                    sMasking = [1]*len(instance.sentence) + [0]*(maxLen-len(instance.sentence))
                else:
                    newSentence = instance.sentence[:maxLen]
                    sMasking = [1]*maxLen
                headMasking = [0]*maxLen
                tailMasking = [0]*maxLen
                try:
                    headMasking [instance.offsetStartHead:min(maxLen,instance.offsetEndHead)] = [1]* (min(maxLen,instance.offsetEndHead) - instance.offsetStartHead)
                except:
                    pass
                try:
                    tailMasking [instance.offsetStartTail:min(maxLen,instance.offsetEndTail)] = [1]* (min(maxLen,instance.offsetEndTail) - instance.offsetStartTail)
                except:
                    pass
                newTriple.append(newSentence)
                sMaskings.append(sMasking)
                headEntityMasking.append (headMasking)
                tailEntityMasking.append (tailMasking)                
                if j >= maxSentenceNum - 1:
                    break
            if len(triple.instance) < maxSentenceNum:
                for j in range (maxSentenceNum-len(triple.instance)):
                    newTriple.append([0]*maxLen)
                    sMaskings.append([0]*maxLen)
                    headEntityMasking.append ([0]*maxLen)
                    tailEntityMasking.append ([0]*maxLen) 
            self.sentenceSamples.append(newTriple)    #sampleNum * sentenceNum * sentenceLen      
            self.sentenceMaskings.append (sMaskings)
            self.headEntityMaskings.append(headEntityMasking)
            self.tailEntityMaskings.append(tailEntityMasking)

    def generateNegSamples(self, changeProba=0.5): #, headSeed=0, tailSeed=0):
        assert changeProba >= 0 and changeProba <= 1.0
        # Generate negtive samples from positive samples
        print("INFO : Generate negtive samples from positive samples.")
        self.negativeSamples = copy.deepcopy(self.positiveSamples)

        # Replacing head or tail
        allSampleNum = range(len(self.positiveSamples))
        for i,sample in enumerate(self.negativeSamples):
            headProbaDistribution = np.random.uniform(low=0.0, high=1.0)
            if headProbaDistribution < changeProba:
                shuffleHead = np.random.choice(allSampleNum)
                self.negativeSamples[i][0] = self.positiveSamples[shuffleHead][0]
            else:
                shuffleTail = np.random.choice(allSampleNum)
                self.negativeSamples[i][2] = self.positiveSamples[shuffleTail][2]
    '''
    Used to transform CSV data to index-form
    ==> csvData : Input CSV data
    ==> repDict : A dict like {column_name : dict(entity_dict)}.
                  The keys are names of the csv columns, the corresponding
                  value is entity/relation dictionary which used to transform
                  entity/realtion to index.
    '''

    def __len__(self):
        return len(self.positiveSamples)

    def __getitem__(self, item):
        if hasattr(self, "negativeSamples"):
            return np.array(self.positiveSamples[item]), np.array(self.negativeSamples[item]),\
                   np.array(self.sentenceSamples[item]), np.array(self.sentenceMaskings[item]),\
                   np.array(self.headEntityMaskings[item]), np.array(self.tailEntityMaskings[item])
        else:
            return np.array(self.positiveSamples[item]), np.array(self.sentenceSamples[item]),\
                   np.array(self.sentenceMaskings[item]), np.array(self.headEntityMaskings[item]),\
                   np.array(self.tailEntityMaskings[item]),