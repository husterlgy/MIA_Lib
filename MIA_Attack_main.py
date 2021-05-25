# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:01:51 2021

@author: user
"""

"""
成员推断攻击的主类: Attacker.
包括了三种常见的成员推断攻击: Shadow Attack, ML-Leaks, Grad-Attack, Output Attack

"""

import pytorch as torch
import sklearn 
import numpy as np
import scipy
import pandas as pd

from shadow_attack import Shadow_Attacker
from ml_leaks import ML_Leaks_Attacker
from output_attack import Output_Attacker


class MIA_Attacker():
    
    def __init__(self, MIA_type):
        if not MIA_type in ['Shadow_Attack','ML_Leaks','Output_Attack']:
            raise TypeError('MIA_type should be a string, within ShadowAttack,MLLeaks,OutputAttack. ')
        
        self.MIA_type = MIA_type
        
        if self.MIA_type == 'Shadow_Attack':
            self.Attacker = Shadow_Attacker()
        elif self.MIA_type == 'ML_Leaks':
            self.Attacker = ML_Leaks_Attacker()
        elif self.MIA_type == 'Output_Attack':
            self.Attacker = Output_Attacker()




