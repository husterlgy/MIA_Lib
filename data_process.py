# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:23:18 2021

@author: user
"""

"""
Data Process: seperate the original data into three part : the training data and testing data for target model , and the training data and testing data for the shadow data. 

There is no overlap between the datasets of the target model and those of the shadow models, but the datasets used for different shadow models can overlap with each other.
 
However, in ML-Leaks, The data each sub-shadow model is trained on is the same. 
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,TensorDataset
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split




class Data_Handler():
    def __init__(self, data_name):
        self.data_name = data_name
        # self.whole_data_tensor = self.Load_Data_Tensor(self.data_name)
        self.whole_data_array = self._Load_Data_Array(self.data_name)
        
        
        self.target_train = None
        self.target_test = None
        self.shadow_train = None
        self.shadow_test = None
        
        self._Target_Shadow_Data_Split(self.whole_data_array)
        
    
    
    
    
    
    
    # Split the data 
    def _Target_Shadow_Data_Split(self, whole_data_array):
        """
        

        Parameters
        ----------
        whole_data_array : Dict = {'data':XX, 'labels':YY}
            All data that used to train the target model and the shadow models.

        Returns
        -------
        target_shadow_data: Dict = {'tranining data of target model':target_train,
                                    'tesing data of target model':target_test,
                                    'tranining data of shadow model':shadow_train,
                                    'tesing data of shadow model':shadow_test,
                                    }
        The splitted data used for target model and shadow models.
        40%->target_train; 10%->target_test
        40%->shadow_train; 10%->shadow_test
        
        """
        N_samples = len(whole_data_array['labels'])
        target_X, shadow_X, target_Y, shadow_Y =train_test_split(whole_data_array['data'],whole_data_array['labels'],test_size=0.5, random_state=0)
        
        target_train_X, target_test_X, target_train_Y, target_test_Y =train_test_split(target_X,target_Y,test_size=0.2, random_state=0)
        
        shadow_train_X, shadow_test_X, shadow_train_Y, shadow_test_Y =train_test_split(shadow_X,shadow_Y,test_size=0.2, random_state=0)
        
        
        # target_shadow_data['target_train'] = {"data": target_train_X, "labels": target_train_Y}
        # target_shadow_data['target_test'] = {"data": target_test_X, "labels": target_test_Y}
        # target_shadow_data['shadow_train'] = {"data": shadow_train_X, "labels": shadow_train_Y}
        # target_shadow_data['shadow_test'] = {"data": shadow_test_X, "labels": shadow_test_Y}
        self.target_train = Data_XY(target_train_X, target_train_Y)
        self.target_test = Data_XY(target_test_X, target_test_Y)
        
        self.shadow_train = Data_XY(shadow_train_X, shadow_train_Y)
        self.shadow_test = Data_XY(shadow_test_X, shadow_test_Y)
        
        
    # Load the original data 
    def _Load_Data_Array(self, data_name):
        """
        

        Parameters
        ----------
        data_name : string
            the name of the used data.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        XX_YY : Dict = {'data':XX, 'labels':YY}
            all data and corresponding labels.

        """
        if not data_name in ['mnist','purchase2','adult','bank']:
            raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')
        if(data_name == 'mnist'):
            trainset = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    
            testset = datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
            train_XX = np.array(trainset.data)
            train_YY = np.array(trainset.targets)
            test_XX = np.array(testset.data)
            test_YY = np.array(testset.targets)
            
            XX = np.vstack([train_XX, test_XX])
            YY = np.hstack([train_YY, test_YY])
            XX = XX.astype(np.float16)/255
            XX = (XX-0.1307)/0.3081
            
        # #model: ResNet-50
        # elif(data_name == 'cifar10'):
        #     transform = transforms.Compose(
        #         [transforms.ToTensor(),
        #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        #     trainset = datasets.CIFAR10(root='./data', train=True,
        #                                             download=True, transform=transform)
            
        #     testset = datasets.CIFAR10(root='./data', train=False,
        #                                             download=True, transform=transform)
        
        #model: 2 FC layers
        elif(data_name == 'purchase2'):
            XX = np.load("./data/purchase/purchase_xx.npy")
            YY = np.load("./data/purchase/purchase_y2.npy")
            

        #model: 2 FC layers
        elif(data_name == 'adult'):
            #load data
            file_path = "./data/adult/"
            data1 = pd.read_csv(file_path + 'adult.data', header=None)
            data2 = pd.read_csv(file_path + 'adult.test', header=None)
            data2 = data2.replace(' <=50K.', ' <=50K')    
            data2 = data2.replace(' >50K.', ' >50K')
            train_num = data1.shape[0]
            data = pd.concat([data1,data2])
           
            #data transform: str->int
            data = np.array(data, dtype=str)
            labels = data[:,14]
            le= LabelEncoder()
            le.fit(labels)
            labels = le.transform(labels)
            data = data[:,:-1]
            
            categorical_features = [1,3,5,6,7,8,9,13]
            # categorical_names = {}
            for feature in categorical_features:
                le = LabelEncoder()
                le.fit(data[:, feature])
                data[:, feature] = le.transform(data[:, feature])
                # categorical_names[feature] = le.classes_
            data = data.astype(float)
            
            n_features = data.shape[1]
            numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
            for feature in numerical_features:
                scaler = MinMaxScaler()
                sacled_data = scaler.fit_transform(data[:,feature].reshape(-1,1))
                data[:,feature] = sacled_data.reshape(-1)
            
            #OneHotLabel
            oh_encoder = ColumnTransformer(
                [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
                remainder='passthrough' )
            oh_data = oh_encoder.fit_transform(data)
            
            xx = oh_data
            yy = labels
            #最终处理，xx进行规范化
            XX = preprocessing.scale(xx)
            YY = np.array(yy)
            
        elif(data_name == 'bank'):
                    #load data
            file_path = "./data/bank/"
            data = pd.read_csv(file_path + 'bank-full.csv',sep=';')
            #data transform
            data = np.array(data, dtype=str)
            labels = data[:,-1]
            le= LabelEncoder()
            le.fit(labels)
            labels = le.transform(labels)
            data = data[:,:-1]
            
            categorical_features = [1,2,3,4,6,7,8,10,15]
            # categorical_names = {}
            for feature in categorical_features:
                le = LabelEncoder()
                le.fit(data[:, feature])
                data[:, feature] = le.transform(data[:, feature])
                # categorical_names[feature] = le.classes_
            data = data.astype(float)
            
            n_features = data.shape[1]
            numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
            for feature in numerical_features:
                scaler = MinMaxScaler()
                sacled_data = scaler.fit_transform(data[:,feature].reshape(-1,1))
                data[:,feature] = sacled_data.reshape(-1)
            #OneHotLabel
            oh_encoder = ColumnTransformer(
                [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
                remainder='passthrough' )
            oh_data = oh_encoder.fit_transform(data)
            XX = oh_data
            YY = labels
            
        # Final combine XX with YY as a Dict
        XX_YY = {"data":XX, "labels":YY}
        
        return XX_YY
    

class Data_XY():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
        
        
        
        
        
        
        
        
        
        