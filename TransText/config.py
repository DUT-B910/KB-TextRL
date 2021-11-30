# -*- coding: utf-8 -*-

'''
Filename : config.py
Function : All the parameter settings used in TransText model, users could change any as they need.
'''

import torch
from code.utils.utils import CheckPath

class Config():
    def __init__(self):
        # Input&Output data arguments
        self.inputPath = './data/Input/'
        self.entity2idFile = './data/Input/entity2id.txt'
        self.relation2idFile = './data/Input/relation2id.txt'
        self.triplesFile = './data/Input/train.txt'
        self.pretrainW2VFile = './data/Input/pretrain_w2v.txt'
        self.pretrainE2VFile = './data/Input/pretrain_e2v.txt'
        self.pretrainR2VFile = './data/Input/pretrain_r2v.txt'
        
        self.pretrainEncoder = "./data/model/TransText_cnn.param"
        self.pretrainModel = "./data/model/TransText.param"

        self.outputPath = './data/Output/'
        self.saveW2VFile = './data/Output/word2vec.txt'
        self.saveE2VFile = './data/Output/entity2vec.txt'
        self.saveR2VFile = './data/Output/relation2vec.txt'
        
        # Middle source path/argument
        self.trainDataPath = './source/train_triples.pkl'
        self.evalDataPath = './source/eval_triples.pkl'
        self.parameterPath = './source/parameters.pkl'
        self.modelPath = "./source/model/"

        # Dataloader arguments
        self.batchSize = 1024
        self.shuffle = True
        self.numWorkers = 0
        self.dropLast = False
        self.repProba = 0.5
        #self.exproba = 0.5

        # paramters:
        self.WORD_EMB_DIM = 100
        self.KG_EMB_DIM = 100
        self.splitRate = 0.1
        self.evaluate = False

        # Model and training general arguments
        self.TransText = {"EmbeddingDim": 100,
                          "KgDim":        100,
                          "Margin":       1.0,
                          "Alpha":        0.01,
                          "L":            2}
        
        # Encoder and training general arguments
        self.lstm = {"hiddenDim":       100,
                     "biDirection":     True}
        self.cnn  = {"hiddenDim":       100,
                     "kernelSizes":     [3,4,5]}
        
        self.maxSentenceLen = 100
        self.maxSentenceNum = 5
        self.usegpu = torch.cuda.is_available()
        self.modelName = "TransText"
        self.encoderName = 'cnn'
        self.optimizer = "Adam"
        self.evalMethod = "MR"
        self.simMeasure = "L2"
        self.modelSaveType = "param"
        self.weightDecay = 0
        self.epochs = 500
        self.evalEpoch = 1
        self.learningRate = 0.01
        self.lrdecay = 0.96
        self.lrdecayEpoch = 5
        self.loadEncoder = False

        # Check Path
        self.CheckPath()

    def CheckPath(self):
        # Check files
        CheckPath(self.entity2idFile)
        CheckPath(self.relation2idFile)
        CheckPath(self.triplesFile)

        # Check dirs
        CheckPath(self.inputPath, raise_error=False)
        CheckPath(self.outputPath, raise_error=False)
        CheckPath(self.modelPath, raise_error=False)


