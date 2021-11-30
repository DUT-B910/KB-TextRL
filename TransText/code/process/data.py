# -*- coding: utf-8 -*-
'''
Filename : Data.py
Function : define the class of Instance & Triple
'''

import re
import os
import json
import codecs
import numpy as np
import pandas as pd
from collections import Counter

class Instance (object):
    def __init__(self,offsetStartHead,offsetEndHead,offsetStartTail,offsetEndTail,sentence):
        self.offsetStartHead = offsetStartHead
        self.offsetEndHead = offsetEndHead
        self.offsetStartTail = offsetStartTail
        self.offsetEndTail = offsetEndTail
        self.sentence = sentence   

class Triple (object):
    def __init__(self,head,tail,relation,instance = []):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.instance = instance     #A list, store a bag of sentences
