# -*- coding :utf-8 -*-
'''
Filename : Train.py
Function : Train the triples
           Stage1: prepare the data
           Stage2: create the model
           Stage3: fit the model
           Stage4: evaluate on the validation set if nesessary (MR)
           Stage5: save the model with the best MR or the last epoch
'''
import os
import json
import torch
import codecs
import pickle
import argparse
import numpy as np
from config import Config
from code.utils import utils
from code.models import TransText
from code.utils import evaluation
from code.dataloader.dataloader import tripleDataset
from code.utils.TensorDevice import LongTensorDevice,FloatTensorDevice,VariableDevice,ParameterDevice
from torch.utils.data import DataLoader

from torch.autograd import Variable
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES']='0'

args = Config()
'''
Generate the dataloader: training, has negative sample generations
'''
def prepareDataloader(args):
    # Initialize dataset and dataloader
    dataset = tripleDataset(dataPath=args.trainDataPath)
                            # entityDictPath=args.entpath,
                            # relationDictPath=args.relpath)
    dataset.paddingSentences(maxLen = args.maxSentenceLen,
                             maxSentenceNum=args.maxSentenceNum)  #padding
    dataset.generateNegSamples(changeProba=args.repProba)
    dataloader = DataLoader(dataset,
                            batch_size=args.batchSize,
                            shuffle=args.shuffle,
                            num_workers=args.numWorkers,
                            drop_last=args.dropLast)
    return dataloader

def prepareEvalDataloader(args):
    dataset = tripleDataset(dataPath=args.evalDataPath)
    dataset.paddingSentences(maxLen = args.maxSentenceLen,
                             maxSentenceNum=args.maxSentenceNum)  #padding
    dataloader = DataLoader(dataset,
                            batch_size=args.batchSize,
                            shuffle=False,
                            drop_last=False)
    return dataloader

'''
cut off the learning rate every i epoch
'''
def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

'''
class of training stage
'''
class trainTriples():
    def __init__(self, args):
        self.args = args

    def prepareData(self):
        print("INFO : Prepare dataloader")
        self.dataloader = prepareDataloader(self.args)
        if self.args.evaluate:
            self.evalloader = prepareEvalDataloader(self.args)

    def prepareModel(self):
        print("INFO : Init model %s"%self.args.modelName)
        if self.args.modelName == "TransText":
            self.model = TransText.TransText(parameterPath = self.args.parameterPath,
                                             encoderName = self.args.encoderName,
                                             embeddingDim=self.args.TransText["EmbeddingDim"],
                                             kgDim = self.args.TransText["KgDim"],
                                             margin=self.args.TransText["Margin"],
                                             L=self.args.TransText["L"],
                                             alpha= self.args.TransText["Alpha"])
        else:
            print("ERROR : No model named %s"%self.args.modelName)
            exit(1)
        if self.args.usegpu:
            self.model.cuda()

    def loadPretrainEncoder(self):
        if self.args.modelName == "TransText":
            print("INFO : Loading pre-training entity and relation embedding!")
            self.model.initialWeight(parameterPath = self.args.parameterPath)
        else:
            print("ERROR : Model %s is not supported!"%self.args.modelName)
            exit(1)

    def loadPretrainModel(self):
        if self.args.modelName == "TransText":
            print("INFO : Loading pre-training model.")
            modelType = os.path.splitext(self.args.premodel)[-1]
            if modelType == ".param":
                self.model.load_state_dict(torch.load(self.args.premodel))
            elif modelType == ".model":
                self.model = torch.load(self.args.premodel)
            else:
                print("ERROR : Model type %s is not supported!")
                exit(1)
        else:
            print("ERROR : Model %s is not supported!" % self.args.modelName)
            exit(1)

    def fit(self):
        EPOCHS = self.args.epochs
        LR = self.args.learningRate
        OPTIMIZER = self.args.optimizer
        if OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         weight_decay=self.args.weightDecay,
                                         lr=LR)
        else:
            print("ERROR : Optimizer %s is not supported."%OPTIMIZER)
            exit(1)

        # Training, GLOBALSTEP and GLOBALEPOCH are used for summary
        minLoss = float("inf")
        bestMR = float("inf")
        GLOBALSTEP = 0
        GLOBALEPOCH = 0
        seed = 2019
        print("INFO : Using seed %d" % seed)
        for epoch in range(EPOCHS):
            GLOBALEPOCH += 1
            STEP = 0
            print("="*20+"EPOCHS(%d/%d)"%(epoch+1, EPOCHS)+"="*20)
            for posX, negX, sentencesX, sentenceMasking, headMasking, tailMasking in self.dataloader:
                # Allocate tensor to devices
                posX = LongTensorDevice (posX, use_cuda = args.usegpu)
                negX = LongTensorDevice (negX,use_cuda = args.usegpu)
                sentencesX = LongTensorDevice (sentencesX,use_cuda = args.usegpu)
                sentenceMasking = LongTensorDevice (sentenceMasking,use_cuda = args.usegpu)
                headMasking = LongTensorDevice (headMasking,use_cuda = args.usegpu)
                tailMasking = LongTensorDevice (tailMasking,use_cuda = args.usegpu)
                
                # Normalize the embedding if neccessary
                # self.model.normalizeEmbedding()
                
                # Calculate the loss from the model
                loss = self.model(posX, negX, sentencesX, sentenceMasking, headMasking, tailMasking)
                if self.args.usegpu:
                    lossVal = loss.cpu().item()
                else:
                    lossVal = loss.item()

                # Calculate the gradient and step down
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print infomation and add to summary
                if minLoss > lossVal:
                    minLoss = lossVal
                print("[TRAIN-EPOCH(%d/%d)-STEP(%d)]Loss:%.4f, minLoss:%f"%(epoch+1, EPOCHS, STEP, lossVal, minLoss))
                STEP += 1
                GLOBALSTEP += 1
            if GLOBALEPOCH % self.args.lrdecayEpoch == 0:
                adjust_learning_rate(optimizer, decay=self.args.lrdecay)
            if GLOBALEPOCH % self.args.evalEpoch == 0 and self.args.evaluate:
                MR = evaluation.MREvaluation(evalloader=self.evalloader,
                                             modelName=self.args.modelName,
                                             simMeasure=args.simMeasure,
                                             model = self.model,
                                             usegpu = self.args.usegpu)
                print("[EVALUATION-EPOCH(%d/%d)]Measure method %s, eval %.4f"% \
                        (epoch+1, EPOCHS, self.args.evalMethod, MR))
                # Save the model if new MR is better
                if MR < bestMR:
                    bestMR = MR
                    self.saveModel()
                    self.dumpEmbedding()
        if not self.args.evaluate:
            self.saveModel()
            self.dumpEmbedding()

    def saveModel(self):
        if self.args.modelSaveType == "param":
            path = os.path.join(self.args.modelPath, "{}_ent{}_rel{}.param".format(self.args.modelName, getattr(self.args, self.args.modelName)["EmbeddingDim"], getattr(self.args, self.args.modelName)["EmbeddingDim"]))
            torch.save(self.model.state_dict(), path)
        elif self.args.modelSaveType == "full":
            path = os.path.join(self.args.modelPath, "{}_ent{}_rel{}.model".format(self.args.modelName, getattr(self.args, self.args.modelName)["EmbeddingDim"], getattr(self.args, self.args.modelName)["EmbeddingDim"]))
            torch.save(self.model, path)
        else:
            print("ERROR : Saving mode %s is not supported!"%self.args.modelSave)
            exit(1)

    def dumpEmbedding(self):
        # save word embedding, entity embedding and relation embedding as txt file
        entWeight = self.model.entityEmbedding.weight.detach().cpu().numpy()
        relWeight = self.model.relationEmbedding.weight.detach().cpu().numpy()
        worWeight = self.model.wordEmbedding.weight.detach().cpu().numpy()
        entityNum, entityDim = entWeight.shape
        relationNum, relationDim = relWeight.shape
        wordNum, wordDim = worWeight.shape
        entsave = os.path.join(self.args.saveE2VFile)
        relsave = os.path.join(self.args.saveR2VFile)
        worsave = os.path.join(self.args.saveW2VFile)
        with codecs.open(entsave, "w", encoding="utf-8") as fp:
            fp.write("{} {}\n".format(entityNum, entityDim))
            for embed in entWeight:
                fp.write("{}\n".format(" ".join(embed.astype(np.str))))
        with codecs.open(relsave, "w", encoding="utf-8") as fp:
            fp.write("{} {}\n".format(relationNum, relationDim))
            for embed in relWeight:
                fp.write("{}\n".format(" ".join(embed.astype(np.str))))
        with codecs.open(worsave, "w", encoding="utf-8") as fp:
            fp.write("{} {}\n".format(wordNum, wordDim))
            for word in self.model.wordDict:
                embed = worWeight[self.model.wordDict[word]]
                fp.write("{} {}\n".format(word," ".join(embed.astype(np.str))))
        # save the encoder parameter
        encodersave = os.path.join(self.args.outputPath, self.args.encoderName + ".param")
        torch.save (self.model.encoder.state_dict(),encodersave)

if __name__ == "__main__":
    # Print args
    utils.printArgs(args)
    
    # Create and train model
    trainModel = trainTriples(args)
    trainModel.prepareData()
    trainModel.prepareModel()
    if args.loadEncoder:
        trainModel.loadPretrainEncoder()
    trainModel.fit()
