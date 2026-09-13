# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import torch
from datetime import datetime
class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.Learning_rate = 1e-4
        self.Epoch = 200
        self.Batch_size =  32
        self.Resume = False
        self.Patience = 20
        self.FC_Dropout = 0.5
        self.test_split = 0.2
        self.validation_split = 0.2
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64

        self.protein_kernel = [4, 6, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 32 # 40
        self.char_dim = 64