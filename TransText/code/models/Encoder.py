# -*- coding: utf-8 -*-

'''
Filename : Encoder.py
Function : Define the sentence encoders
           Input: a bag of sentences of a triple
           Output: a feature of sentences with size of batch_size*kgDim*1 
                   a feature of head entity with size of batch_size*kgDim*1 
                   a feature of tail entity with size of batch_size*kgDim*1 
'''

import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from code.utils.TensorDevice import LongTensorDevice,FloatTensorDevice,VariableDevice,ParameterDevice

np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
mySeed = np.random.RandomState(2019)

'''
encoder: lstm
         steps1: Every sentences pass through an LSTM encoder
         steps2: Use maxpooling to catch each sentence feature
         steps3: Use meanpooling to catch the feature of all the sentences
         steps4: Average all the entity words in sentences to generate entity feature 
'''
class lstmEncoder(nn.Module):
    def __init__(self, embeddingDim,hiddenDim,kgDim,biDirection, usegpu =True):
        super(lstmEncoder, self).__init__()
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.kgDim = kgDim
        self.usegpu = usegpu
        self.lstm  = nn.LSTM(input_size = embeddingDim, hidden_size = hiddenDim, bidirectional =biDirection,dropout = 0.5, batch_first = True)
        self.linearW = ParameterDevice(mySeed.uniform(-0.01, 0.01, ( self.hiddenDim * 2, self.kgDim )),use_cuda=usegpu, requires_grad = True)
        self.linearb = ParameterDevice(mySeed.uniform(-0.01, 0.01, ( 1, self.kgDim )),use_cuda=usegpu, requires_grad = True)
    
    def forward (self, sentences,sentenceMasking,headMasking,tailMasking):
        batchSize = sentences.size()[0]
        sentenceLen = sentences.size()[2]
        sentenceNum = sentences.size()[1]
        sentences = sentences.view(-1,sentenceLen,self.embeddingDim)  #batchsize*sentenceNum*maxlen*inputDim   ->  (batchsize*sentenceNum)*maxlen*inputDim
        outputSentences,_ = self.lstm (sentences)                         #(batchsize*sentenceNum)*maxlen*(hiddenDim*2)
        outputSentences = outputSentences.view(batchSize,sentenceNum,sentenceLen,self.hiddenDim*2)  #batchsize*sentenceNum*maxlen*(hiddenDim*2)
        outputSentences = torch.mul(outputSentences,sentenceMasking.unsqueeze(-1))      #masking the padding words
        #print (outputSentences.size())
        
        # cut out the head entity
        headTotalLength = torch.sum(headMasking,dim = (1,2)).unsqueeze(1) # batchsize*1
        zeroTensor = torch.eq (headTotalLength,LongTensorDevice(Tensor = np.zeros(headTotalLength.size()), use_cuda = self.usegpu))
        headTotalLength = headTotalLength + zeroTensor
        outputHead = torch.sum(torch.mul(outputSentences,headMasking.unsqueeze(-1)),dim = (1,2)) / headTotalLength   #batchsize*(hiddenDim*2)
        headFeature = outputHead.unsqueeze(1)                        #batchsize*1*(hiddenDim*2)
        finalHeadFeature = torch.bmm(headFeature, self.linearW.unsqueeze(0).expand(batchSize,-1,-1)) + self.linearb.unsqueeze(0).expand(batchSize,-1,-1)
        
        # cut out the tail entity
        tailTotalLength = torch.sum(tailMasking,dim = (1,2)).unsqueeze(1)
        zeroTensor = torch.eq (tailTotalLength,LongTensorDevice(Tensor = np.zeros(tailTotalLength.size()), use_cuda = self.usegpu))
        tailTotalLength = tailTotalLength + zeroTensor
        outputTail = torch.sum(torch.mul(outputSentences,tailMasking.unsqueeze(-1)),dim = (1,2)) / tailTotalLength        
        tailFeature = outputTail.unsqueeze(1)  
        finalTailFeature = torch.bmm(tailFeature, self.linearW.unsqueeze(0).expand(batchSize,-1,-1)) + self.linearb.unsqueeze(0).expand(batchSize,-1,-1)

        featureSentences = F.max_pool2d (outputSentences.transpose(1,3),kernel_size = (sentenceLen,1))    #batchSize*(hiddenDim*2)*1*sentenceNum
        # featureSentence = F.avg_pool1d (featureSentences,kernel_size = sentenceNum).transpose(1,2)     #batchSize*1*(hiddenDim*2)
        realSentenceNum = torch.sum(sentenceMasking[:,:,0],dim = 1).unsqueeze(1).unsqueeze(1).expand(batchSize,1,self.hiddenDim*2)
        featureSentence = torch.sum(featureSentences,dim = 3).transpose(1,2)/realSentenceNum                     #batchSize*1*(hiddenDim*2)
        finalFeature = torch.bmm(featureSentence, self.linearW.unsqueeze(0).expand(batchSize,-1,-1)) + self.linearb.unsqueeze(0).expand(batchSize,-1,-1) #batchSize*1*(hiddenDim*2)
        
        return finalFeature.squeeze(1),finalHeadFeature.squeeze(1),finalTailFeature.squeeze(1)

'''
encoder: lstm
         steps1: Every sentences pass through an CNN encoder
         steps2: Use maxpooling to catch each sentence feature
         steps3: Use meanpooling to catch the feature of all the sentences
         steps4: Average all the entity words in sentences to generate entity feature 
'''
class cnnEncoder(nn.Module):
    def __init__(self, embeddingDim,hiddenDim,kgDim,kernelSizes,usegpu =True):
        super(cnnEncoder, self).__init__()
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.kgDim = kgDim
        self.kernelSizes = kernelSizes
        self.usegpu = usegpu
        self.cnnList = nn.ModuleList([nn.Conv1d(embeddingDim,hiddenDim, kernel_size = size,padding = int((size-1)/2)) for size in kernelSizes])
        self.linearW = ParameterDevice(mySeed.uniform(-0.01, 0.01, ( self.hiddenDim * len(kernelSizes), self.kgDim )),use_cuda=usegpu, requires_grad = True)
        self.linearb = ParameterDevice(mySeed.uniform(-0.01, 0.01, ( 1, self.kgDim )),use_cuda=usegpu, requires_grad = True)
    
    def forward (self, sentences,sentenceMasking,headMasking,tailMasking):
        batchSize = sentences.size()[0]
        sentenceLen = sentences.size()[2]
        sentenceNum = sentences.size()[1]
        sentences = sentences.view(-1,sentenceLen,self.embeddingDim).transpose(1,2)  #batchsize*sentenceNum*maxlen*inputDim   ->  (batchsize*sentenceNum)*inputDim*maxlen
        outputSentences = []
        for kernel,cnn in zip(self.kernelSizes,self.cnnList):                         #(batchsize*sentenceNum)*maxlen*(hiddenDim*2)
            if not kernel%2:
                outputSentence = cnn(torch.cat([sentences,VariableDevice(Tensor = np.zeros((batchSize*sentenceNum,self.embeddingDim,1)),requires_grad = False,use_cuda = self.usegpu)],2))
            else:
                outputSentence = cnn(sentences)
            outputSentences.append(outputSentence)
        outputSentences = torch.cat(outputSentences,1)
        outputSentences = outputSentences.view(batchSize,sentenceNum,self.hiddenDim * len(self.kernelSizes),sentenceLen).transpose(2,3)  #batchsize*sentenceNum*maxlen*(hiddenDim*2)
        outputSentences = torch.mul(outputSentences,sentenceMasking.unsqueeze(-1))      #masking the padding words
        #print (outputSentences.size())
        
        # cut out the head entity
        headTotalLength = torch.sum(headMasking,dim = (1,2)).unsqueeze(1) # batchsize*1
        zeroTensor = torch.eq (headTotalLength,LongTensorDevice(Tensor = np.zeros(headTotalLength.size()), use_cuda = self.usegpu))
        headTotalLength = headTotalLength + zeroTensor
        outputHead = torch.sum(torch.mul(outputSentences,headMasking.unsqueeze(-1)),dim = (1,2)) / headTotalLength   #batchsize*(hiddenDim*2)
        headFeature = outputHead.unsqueeze(1)                        #batchsize*1*(hiddenDim*2)
        finalHeadFeature = torch.bmm(headFeature, self.linearW.unsqueeze(0).expand(batchSize,-1,-1)) + self.linearb.unsqueeze(0).expand(batchSize,-1,-1)
        
        # cut out the tail entity
        tailTotalLength = torch.sum(tailMasking,dim = (1,2)).unsqueeze(1)
        zeroTensor = torch.eq (tailTotalLength,LongTensorDevice(Tensor = np.zeros(tailTotalLength.size()), use_cuda = self.usegpu))
        tailTotalLength = tailTotalLength + zeroTensor
        outputTail = torch.sum(torch.mul(outputSentences,tailMasking.unsqueeze(-1)),dim = (1,2)) / tailTotalLength        
        tailFeature = outputTail.unsqueeze(1)  
        finalTailFeature = torch.bmm(tailFeature, self.linearW.unsqueeze(0).expand(batchSize,-1,-1)) + self.linearb.unsqueeze(0).expand(batchSize,-1,-1)

        featureSentences = F.max_pool2d (outputSentences.transpose(1,3),kernel_size = (sentenceLen,1))    #batchSize*(hiddenDim*2)*1*sentenceNum
        # featureSentence = F.avg_pool1d (featureSentences,kernel_size = sentenceNum).transpose(1,2)     #batchSize*1*(hiddenDim*2)
        realSentenceNum = torch.sum(sentenceMasking[:,:,0],dim = 1).unsqueeze(1).unsqueeze(1).expand(batchSize,1,self.hiddenDim* len(self.kernelSizes))
        featureSentence = torch.sum(featureSentences,dim = 3).transpose(1,2)/realSentenceNum                     #batchSize*1*(hiddenDim*2)
        finalFeature = torch.bmm(featureSentence, self.linearW.unsqueeze(0).expand(batchSize,-1,-1)) + self.linearb.unsqueeze(0).expand(batchSize,-1,-1) #batchSize*1*(hiddenDim*2)
        
        return finalFeature.squeeze(1),finalHeadFeature.squeeze(1),finalTailFeature.squeeze(1)