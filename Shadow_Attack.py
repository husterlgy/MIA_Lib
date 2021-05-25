# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:08:18 2021

@author: user
"""

"""
Shadow Attack
INPUT：
        the training function of target ML model；
        A Shadow Dataset that has the similar distribution with traget model's training data 
        the prediction interface 

OUTPUT：
"""

import scipy
import numpy as np
import sklearn 
import pandas as pd


class Shadow_Attacker():
    
    def __init__(self, 
                 N_shadow_model=10, 
                 N_attack_model=10, 
                 N_train_samples, 
                 target_model_trainer, 
                 targe_shadow_data
                 ):
        """
        

        Parameters
        ----------
        N_shadow_model : Int
            the number of Shadow  models .
        N_attack_model : Int
            the number of attack models.
        N_train_samples : Int
            the number of selected samples used to train the target models.
        target_model_trainer : base.Model_Trainer
            the trainer (or the training processing ) of the target model, which can be used to train the shadow model.
        targe_shadow_data : data_process.Data_Handler
            it includes the training data and testing data for the target model and shadow model .

        Returns
        -------
        None.

        """
        self.N_shadow_model = N_shadow_model
        self.N_attack_model = N_attack_model
        self.N_train_samples = N_train_samples
        
        self.target_model_trainer = target_model_trainer
        self.data = targe_shadow_data
        
        #initial the list of shadow models and attack models 
        self.N_shadow_model = []
        self.N_attack_model = []
        
        #train the shadow models with the same setting of target model 
        self._Train_Shadow_Model(self.data)
        #train the attack models 
        self._Train_Attack_Model(self.data)
        
    def _Train_Shadow_Model(self):
        for ii in range(self.N_shadow_model):
            pass
    
    def _Train_Attack_Model(self):
        for ii in range(self.N_attack_model):
            pass
    
    def Attack(self):
        pass

