# -*- coding: utf-8 -*-
'''
Filename : evaluation.py
Function : When the args.evaluation is true, use this file to evaluate the model in current epoch.
           Input: evaluation dataset
           Output: evaluation score ==> MR
'''


from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import dataloader
from torch.autograd import Variable
import copy
from code.utils.TensorDevice import LongTensorDevice,FloatTensorDevice,VariableDevice,ParameterDevice

'''
Evaluation for TransText
'''
def evalTransText(triples, sentences,sentenceMasking,headMasking,tailMasking,simMeasure, model,usegpu):
    triples = triples.numpy()
    sampleNum = triples.shape[0]
    allRelationNum = len(model.relationDict)
    rank = 0
    triplesX = LongTensorDevice (triples, use_cuda = usegpu)
    sentencesX = LongTensorDevice (sentences,use_cuda = usegpu)
    sentenceMasking = LongTensorDevice (sentenceMasking,use_cuda = usegpu)
    headMasking = LongTensorDevice (headMasking,use_cuda = usegpu)
    tailMasking = LongTensorDevice (tailMasking,use_cuda = usegpu)
    realScores = model.evalForward (triplesX,sentencesX, sentenceMasking,headMasking,tailMasking).detach().cpu().numpy()   # N*1

    for entity in range(allRelationNum):
        candidateEntity = np.array([entity]*sampleNum)
        candidateTriples = copy.deepcopy(triples)        # N*3
        candidateTriples[:,2] = candidateEntity
        triplesX = LongTensorDevice (candidateTriples, use_cuda = usegpu)
        candidateScore = model.evalForward (triplesX,sentencesX,sentenceMasking,headMasking,tailMasking)
        candidateScore = candidateScore.detach().cpu().numpy()
        judgeMatrix = candidateScore-realScores
        if simMeasure == "L2" or simMeasure == "L1":
            judgeMatrix[judgeMatrix > 0] = 0
            judgeMatrix[judgeMatrix < 0] = 1
        elif simMeasure == "dot" or simMeasure == "cos":
            judgeMatrix[judgeMatrix > 0] = 1
            judgeMatrix[judgeMatrix < 0] = 0
        else:
            print("ERROR : Similarity measure is not supported!")
            exit(1)
        rank += np.sum(judgeMatrix)
    return rank, sampleNum

'''
Now, only MR metric is available
'''
def MREvaluation(evalloader:dataloader, modelName, model, simMeasure="L2", usegpu = True):
    R = 0
    N = 0
    for triple,sentences,sentenceMasking,headMasking,tailMasking in evalloader:
        if modelName == "TransText":
            r,n = evalTransText(triple,sentences,sentenceMasking,headMasking,tailMasking, simMeasure, model,usegpu)
        else:
            print("ERROR : The %s evaluation is not supported!" % model)
            exit(1)
        R += r    #sum of rank
        N += n     #sum of the instance
    return (R / N)              #MeanRank



