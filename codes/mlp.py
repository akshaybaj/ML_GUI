
from PyQt5.QtWidgets import *
import sys,os,re,pickle
import data_visualise,common,add_steps

from PyQt5 import uic, QtWidgets ,QtCore, QtGui

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class UI(QMainWindow):
    def __init__(self,df,target,user_actions):
        super(UI, self).__init__()
        uic.loadUi("../ui_files/MLP.ui", self)
        self.user_act=user_actions
        global data ,steps
        data=data_visualise.data_()
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        self.test_size= self.findChild(QLabel,"test_size") 
               
        self.train_btn=self.findChild(QPushButton,"train")
        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.train_btn.clicked.connect(self.training)
        self.test_size_btn.clicked.connect(self.test_split)
        self.reshape_btn.clicked.connect(self.reshape_data)
        self.go.clicked.connect(self.create_model)
        self.conf_mat.clicked.connect(self.conf_matrix)
        self.dwnld.clicked.connect(self.download_model)
        self.setvalue()
        
        self.show()

    def setvalue(self):
        self.target.setText(self.target_value)
        self.columns.clear()
        self.columns.addItems(self.column_list)
        self.data_shape.setText(str(self.df.shape))
         
        self.reshape.setText(re.sub('[()]', '', str(self.df.shape)))

       
     
    def download_model(self):

        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")
        #file = open(name[0],'w')
        
        pkl_filename = name[0]
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.mlp, file)  
        
        self.user_act.save_file(pkl_filename)

    def reshape_data(self):
        
        shape=str(self.reshape.text())
        self.df=np.array(self.df)
        self.df=self.df.reshape(eval(shape))
        self.after_reshape.setText(str(self.df.shape))
            
    
    def test_split(self):

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=0)
        print(self.y_train.shape)
        print(self.y_test.shape)
        print(self.y_train.shape)
        self.train_size.setText(str(self.x_train.shape))
        self.test_size.setText(str(self.x_test.shape))

    def create_model(self):

        self.hidden_layer=str(self.mlp_layers.text())
        self.active_=self.activations.currentText()
        self.solver_=self.solvers.currentText()
        self.alpha_=float(self.alpha_val.text())
        self.lr=float(self.learning_rate.text())
        self.max_iter_=int(self.iteration.text())
    
    def training(self):

        self.mlp = MLPClassifier(hidden_layer_sizes=eval(self.hidden_layer), activation=self.active_, learning_rate_init=self.lr,alpha=self.alpha_,max_iter=self.max_iter_,random_state=1,verbose=True)
        
        
        original=sys.stdout
        sys.stdout = open('summary.txt', 'w')
        print(self.mlp.fit(self.x_train,self.y_train))
        sys.stdout=original
        text=open('summary.txt').read()
        self.summary.setPlainText(text)
        os.remove('summary.txt')
        
        
        self.pre=self.mlp.predict(self.x_test)
        self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))
        self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))
        self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))
        self.accuracy.setText(str(metrics.accuracy_score(self.y_test,self.pre)))

        text=steps.classification_(self.y_test,self.pre)
        self.report.setPlainText(text)
    def conf_matrix(self):

        data = {'y_Actual':self.y_test,'y_Predicted':self.pre }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()

    