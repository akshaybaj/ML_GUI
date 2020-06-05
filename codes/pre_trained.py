
from PyQt5.QtWidgets import *
import sys,os,re,pickle
import data_visualise,common

from PyQt5 import uic, QtWidgets ,QtCore, QtGui

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class UI(QMainWindow):
    def __init__(self,df,target,pickle_model,path):
        super(UI, self).__init__()
        uic.loadUi("../ui_files/pre_trained.ui", self)
        self.path=path
        global data 
        data=data_visualise.data_()
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        self.test_size= self.findChild(QLabel,"test_size") 
        self.model=pickle_model       
        

        self.conf_mat.clicked.connect(self.conf_matrix)
        self.test.clicked.connect(self.test_model)
        self.setvalue()
        self.show()

    def setvalue(self):
        self.target.setText(self.target_value)
        self.columns.clear()
        self.columns.addItems(self.column_list)
        self.data_shape.setText(str(self.df.shape))
        
        original=sys.stdout
        sys.stdout = open('summary.txt', 'w')
        print(self.model)
        sys.stdout=original
        text=open('summary.txt').read()
        self.model_details.setPlainText(text)
        os.remove('summary.txt')

        x=self.path.split(".")
        
        text=open(x[0]+".txt").read()
        self.user_actions.setPlainText(text)
        


    def conf_matrix(self):

        data = {'y_Actual':self.X[self.target_value],'y_Predicted':self.pre }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()

    def test_model(self):

        
        self.pre=self.model.predict(self.df)
        self.mae.setText(str(metrics.mean_absolute_error(self.X[self.target_value],self.pre)))
        self.mse.setText(str(metrics.mean_squared_error(self.X[self.target_value],self.pre)))
        self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.X[self.target_value],self.pre))))
        self.accuracy.setText(str(metrics.accuracy_score(self.X[self.target_value],self.pre)))
    