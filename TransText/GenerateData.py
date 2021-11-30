# -*- coding:utf-8 -*-
'''
Filename : GenerateData.py
Function : Package all the data into a pkl file
           Pacakge all the parameters/embeddings into a pkl file
'''

from code.process.LoadData import loadTerm2id, loadTrainTriples, loadPretrainW2V, loadPretrainE2V, splitData
import pickle
from config import Config

if __name__ == '__main__':
    
    args = Config()
    # step 1 Transform raw input to standard format
    # track 1 load entity2id
    print ('INFO : loading entity2id...')
    entityDict = loadTerm2id (dataPath = args.entity2idFile)
    
    # track 2 load relation2id
    print ('INFO : loading relation2id...')
    relationDict = loadTerm2id (dataPath = args.relation2idFile)
    
    # track 3 load Triples and translate them into ID for training
    print ('INFO : loading all training triples...')
    allTriples,wordDict = loadTrainTriples (
        dataPath = args.triplesFile,
        entityDict = entityDict,
        relationDict = relationDict,
        wordDict = {}
        )
    
    # track 4 load pretrained word2vec
    print ('INFO : loading pretrained word2vec...')
    wordEmbedding = loadPretrainW2V(
        embedPath= args.pretrainW2VFile,
        wordDict = wordDict,
        embeddingDim = args.WORD_EMB_DIM
        )

    # track 5 load pretrained entity2vec
    print ('INFO : loading pretrained entity2vec...')
    entityEmbedding = loadPretrainE2V(
        embedPath = args.pretrainE2VFile,
        entityDict = entityDict,
        embeddingDim = args.KG_EMB_DIM
        )

    # track 6 load pretrained relation2vec
    print ('INFO : loading pretrained relation2vec...')
    relationEmbedding = loadPretrainE2V(
        embedPath = args.pretrainR2VFile,
        entityDict = relationDict,
        embeddingDim = args.KG_EMB_DIM
        )
    
    #step 2 Split Data:
    
    if args.evaluate:
        print ('INFO : spliting data into training set and testing set...')
        allTriples,evalTriples = splitData(
            allTriples = allTriples,
            splitRate = args.splitRate)
        pickle.dump (evalTriples,open(args.evalDataPath,'wb'))
    
    print ('INFO : dumping all the preprocessing data...')
    #step 3 Dump all the file into PKL
    pickle.dump (allTriples,open(args.trainDataPath,'wb'))
    pickle.dump ([wordEmbedding,entityEmbedding,relationEmbedding,wordDict,entityDict,relationDict],open(args.parameterPath, 'wb'))