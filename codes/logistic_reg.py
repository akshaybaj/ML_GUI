
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys ,pickle
import data_visualise
import table_display
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import common



class UI(QMainWindow):
    def __init__(self,df,target,user_actions):
        super(UI, self).__init__()
        uic.loadUi("../ui_files/LogisticRegression.ui", self)
        self.user_act=user_actions
        global data ,steps
        data=data_visualise.data_()
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        self.test_size= self.findChild(QLabel,"test_size") 
        self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        self.test_size= self.findChild(QLabel,"test_size")  
        
        self.c_=self.findChild(QLineEdit,"c_")
        self.penalty=self.findChild(QComboBox,"penalty")
        self.solver=self.findChild(QComboBox,"solver")        
        self.dual=self.findChild(QComboBox,"dual")       
        self.max_iter=self.findChild(QLineEdit,"max_iter")
        self.fit_inter=self.findChild(QComboBox,"fit_inter")  
        self.multi_class=self.findChild(QComboBox,"multi_class")
        self.tol=self.findChild(QLineEdit,"tol")
        self.train_btn=self.findChild(QPushButton,"train")
        
        self.mae=self.findChild(QLabel,"mae")
        self.mse=self.findChild(QLabel,"mse")
        self.rmse=self.findChild(QLabel,"rmse")
        self.accuracy=self.findChild(QLabel,"accuracy")
        self.roc_btn=self.findChild(QPushButton,"roc")
        self.X_combo=self.findChild(QComboBox,"X_combo")
        self.Y_combo=self.findChild(QComboBox,"Y_combo")

        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.train_btn.clicked.connect(self.training)
        self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
        #self.roc_btn.clicked.connect(self.roc_plot)
        self.conf_mat_btn.clicked.connect(self.conf_matrix)
        self.test_size_btn.clicked.connect(self.test_split)
        self.dwnld.clicked.connect(self.download_model)
        self.setvalue()
        self.show()

    def setvalue(self):
        self.target.setText(self.target_value)
        self.columns.clear()
        self.columns.addItems(self.column_list)
        self.X_combo.addItems(self.column_list)
        self.Y_combo.addItems(self.column_list)

    
    def test_split(self):

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=0)
        print(self.y_train.shape)
        print(self.y_test.shape)
        self.train_size.setText(str(self.x_train.shape))
        self.test_size.setText(str(self.x_test.shape))

    def download_model(self):

        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")
        #file = open(name[0],'w')
        
        pkl_filename = name[0]
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.lr, file)  
        
        self.user_act.save_file(pkl_filename)  

    def training(self):

        self.lr = LogisticRegression(C=float(self.c_.text()),penalty=self.penalty.currentText(),dual=self.dual.currentText()=='True',tol=float(self.tol.text()),max_iter=float(self.max_iter.text()),fit_intercept=self.fit_inter.currentText()=='True',random_state=1,solver=self.solver.currentText(),multi_class=self.multi_class.currentText())
        self.lr.fit(self.x_train,self.y_train)
        
        self.pre=self.lr.predict(self.x_test)
        self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))
        self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))
        self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))
        self.accuracy.setText(str(accuracy_score(self.pre,self.y_test)))
        text=steps.classification_(self.y_test,self.pre)
        self.report.setPlainText(text)

    def conf_matrix(self):

        data = {'y_Actual':self.y_test.values,'y_Predicted':self.pre }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()

    