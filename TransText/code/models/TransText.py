# -*- coding: utf-8 -*-

'''
Filename : TransText.py
Function : Define the knowledge representation learning model
           Input: Positive triples in knowledge base
                  Generated negative triples
                  A bag of sentences for each triple
                  Sentences&entities masking of each triple 
'''

import torch
import codecs
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from config import Config

from code.models.Encoder import cnnEncoder,lstmEncoder
from code.utils.TensorDevice import FloatTensorDevice,LongTensorDevice,VariableDevice,ParameterDevice

args = Config()
class TransText(nn.Module):
    def __init__(self, parameterPath, encoderName, embeddingDim,kgDim,usegpu = True, margin=1.0, L=2,alpha = 0.01):
        super(TransText, self).__init__()
        assert (L == 1 or L == 2)
        self.model = "TransText"
        self.margin = margin
        self.L = L
        self.alpha = alpha
        self.usegpu = usegpu
        wordEmbedding,entityEmbedding,relationEmbedding,wordDict,entityDict,relationDict = pickle.load (open(parameterPath,'rb'),encoding = 'iso-8859-1')
        self.wordDict = wordDict
        self.entityDict = entityDict
        self.relationDict = relationDict
        self.entityEmbedding = nn.Embedding(num_embeddings=entityEmbedding.shape[0],
                                            embedding_dim=embeddingDim)
        self.entityEmbedding.weight = ParameterDevice(Tensor=entityEmbedding,
                                                      requires_grad = True,
                                                      use_cuda =usegpu)
        self.relationEmbedding = nn.Embedding(num_embeddings=relationEmbedding.shape[0],
                                              embedding_dim=embeddingDim)
        self.relationEmbedding.weight = ParameterDevice(Tensor=relationEmbedding,
                                                        requires_grad = True,
                                                        use_cuda =usegpu)
        self.wordEmbedding = nn.Embedding(num_embeddings=wordEmbedding.shape[0],
                                          embedding_dim=embeddingDim)
        self.wordEmbedding.weight = ParameterDevice(Tensor=wordEmbedding,
                                                    requires_grad = True,
                                                    use_cuda =usegpu)  
        self.distfn = nn.PairwiseDistance(L)
        self.mseLoss = nn.MSELoss()
        if encoderName == 'lstm':
            self.encoder = lstmEncoder (embeddingDim = embeddingDim,
                                        biDirection = args.lstm['biDirection'],
                                        hiddenDim = args.lstm['hiddenDim'],
                                        kgDim = kgDim,
                                        usegpu = usegpu)
        elif encoderName == 'cnn':
            self.encoder = cnnEncoder (embeddingDim = embeddingDim,
                                       kernelSizes = args.cnn['kernelSizes'],
                                       hiddenDim = args.cnn['hiddenDim'],
                                       kgDim = kgDim,
                                       usegpu = usegpu)
        else:
            print("ERROR : No encoder named %s"%encoderName)
            exit(1)

    '''
    This function used to calculate score, steps follows:
    Step1: Split input as head, relation and tail index column
    Step2: Transform index tensor to embedding tensor
    Step3: Calculate MSE loss of text encoded entity/relation embeddings and triple encoded entity/relation embeddings
    Step4: Calculate distance as final score
    '''
    def scoreOp(self, inputTriple, sentenceFeature,headEntityFeature = None,tailEntityFeature = None):
        # Step1
        # head : shape(batch_size, 1)
        # relation : shape(batch_size, 1)
        # tail : shape(batch_size, 1)
        batchSize = inputTriple.size()[0]
        head, relation, tail = torch.chunk(input=inputTriple,
                                           chunks=3,
                                           dim=1)
        # Step2
        # head : shape(batch_size, 1, embedDim)
        # relation : shape(batch_size, 1, embedDim)
        # tail : shape(batch_size, 1, embedDim)
        head = torch.squeeze(self.entityEmbedding(head), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)
        textRelation = torch.squeeze(sentenceFeature, dim=1)
        
        # Step 3
        # Calculate loss of the alignment
        # if there is no entity (cut off), replace it with an all-zero embeddings
        try:
            outputHead = self.mseLoss(head,headEntityFeature)
        except:
            outputHead = VariableDevice(Tensor = np.zeros(batchSize),use_cuda = self.usegpu,requires_grad = False)
        
        try:
            outputTail = self.mseLoss(tail,tailEntityFeature)
        except:
            outputTail = VariableDevice(Tensor = np.zeros(batchSize),use_cuda = self.usegpu,requires_grad = False)
        
        outputRelation = self.mseLoss(relation,textRelation)

        # Step4 and Step4
        # output : shape(batch_size, embedDim) ==> shape(batch_size, 1)
        outputEntity = self.distfn(head+relation, tail)
        return outputEntity, outputHead,outputTail, outputRelation

    '''
    In every training epoch, the entity embedding could be normalize
    Step1: Get numpy.array from embedding weight
    Step2: Normalize array
    Step3: Assign normalized array to embedding
    '''
    def normalizeEmbedding(self):
        embedWeight = self.entityEmbedding.weight.detach().cpu().numpy()
        embedWeight = embedWeight / np.sqrt(np.sum(np.square(embedWeight), axis=1, keepdims=True))
        self.entityEmbedding.weight.data.copy_(torch.from_numpy(embedWeight))

        # embedWeight = self.relationEmbedding.weight.detach().cpu().numpy()
        # embedWeight = embedWeight / np.sqrt(np.sum(np.square(embedWeight), axis=1, keepdims=True))
        # self.relationEmbedding.weight.data.copy_(torch.from_numpy(embedWeight))

    '''
    Input:
    posX : (torch.tensor)The positive triples tensor, shape(batch_size, 3)
    negX : (torch.tensor)The negtive triples tensor, shape(batch_size, 3)
    posSentence : (torch.tensor)The sentences tensor, shape(batch_size, sentence_num,max_len)
    sentenceMasking : (torch.tensor)The sentence Masking tensor, where the padding words are 0, shape(batch_size, sentence_num,max_len)
    headMasking : (torch.tensor)The head entity masking tensor, where the words in head entity are 1, shape(batch_size, sentence_num,max_len)
    tailMasking : (torch.tensor)The tail entity masking tensor, where the words in tail entity are 1, shape(batch_size, sentence_num,max_len)
    '''
    def forward(self, posX,  negX, posSentence, sentenceMasking, headMasking, tailMasking):
        size = posX.size()[0]
        # Calculate score
        posSentence = self.wordEmbedding(posSentence)
        sentenceFeature,headEntityFeature,tailEntityFeature = self.encoder(posSentence,sentenceMasking,headMasking,tailMasking)
        posEntityScore, headScoreLoss, tailScoreLoss, relationScoreLoss = self.scoreOp(posX,sentenceFeature,headEntityFeature,tailEntityFeature)
        negEntityScore, _, _,_ = self.scoreOp(negX,sentenceFeature)

        # Get margin ranking loss
        lossEntity = torch.sum(F.relu(input=posEntityScore-negEntityScore+self.margin))/size
        lossHead = torch.sum(headScoreLoss)/size
        lossTail = torch.sum(tailScoreLoss)/size
        lossRelation = torch.sum(relationScoreLoss)/size
        return lossEntity + (lossHead + lossTail + lossRelation) * self.alpha
    
    '''
    Evaluation forward:
    without negative instance,calculate the distance between entity1&relation&entity2, igonore the alignment loss
    '''
    def evalForward(self, posX, posSentence, sentenceMasking, headMasking, tailMasking):
        size = posX.size()[0]
        # Calculate score
        posSentence = self.wordEmbedding(posSentence)
        sentenceFeature,headEntityFeature,tailEntityFeature = self.encoder(posSentence,sentenceMasking,headMasking,tailMasking)
        posEntityScore, _,_,_ = self.scoreOp(posX,sentenceFeature,headEntityFeature,tailEntityFeature)

        # Get margin ranking loss
        return posEntityScore
    
    '''
    Used to load pretraining encoder.
    '''
    def initialWeight(self, encoderPath):
        print("INFO : Loading pretrained encoder.")
        self.encoder.load_state_dict(torch.load(encoderPath))
