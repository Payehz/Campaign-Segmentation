# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:46:42 2022

@author: User
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization

class EDA():
    def __init__(self):
        pass
    
    def cramers(self,confusionmatrix):
        '''
        

        Parameters
        ----------
        confusionmatrix : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        chi2 = ss.chi2_contingency(confusionmatrix)[0]
        n = confusionmatrix.sum()
        phi2 = chi2/n
        r,k = confusionmatrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2/(n-1))
        kcorr = k - ((k-1)**2/(n-1))
        return np.sqrt(phi2corr / min((kcorr-1),(rcorr-1)))
    
    def con_graph_plot(self,df,con):
        '''
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        con : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.figure()
        sns.distplot(df[con])
        plt.show()
        
    def cat_graph_plot(self,df,cat):
        '''
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        cat : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.figure()
        sns.countplot(df[cat])
        plt.show()


class ModelCreation():
    def __init__(self):
        pass
    
    def simple_two_layer_model(self,num_class,num_feature, drop_rate=0.2,node_num=64):
        
        '''
        Parameters    
        ----------    
        nb_class : TYPE        
            DESCRIPTION.    
        drop_rate : TYPE, optional       
            DESCRIPTION. The default is 0.2.    
        node_num : TYPE, optional        
            DESCRIPTION. The default is 32.   
        Returns    
        -------   
        classifier : TYPE        
            DESCRIPTION.
        '''
        model = Sequential() # To create a container
        model.add(Input(shape=num_feature)) # To include an input layer
        model.add(Flatten())
        model.add(Dense(node_num,activation='sigmoid',name='HiddenLayer1')) # Hidden layer 1
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(node_num,activation='sigmoid',name='HiddenLayer2')) # Hidden layer 2
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(num_class,activation='softmax',name='OutputLayer')) # output layer
        
        return model
    
    def result_plot(self,tr_loss,val_loss,tr_acc,val_acc):
        '''
        

        Parameters
        ----------
        tr_loss : TYPE
            DESCRIPTION.
        val_loss : TYPE
            DESCRIPTION.
        tr_acc : TYPE
            DESCRIPTION.
        val_acc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(tr_loss)
        plt.plot(val_loss)
        plt.legend(['train_loss', 'val_loss'])
        plt.show()

        plt.figure()
        plt.plot(tr_acc)
        plt.plot(val_acc)
        plt.legend(['train_acc', 'val_acc'])
        plt.show()


















