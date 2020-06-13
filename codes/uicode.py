from PyQt5.QtWidgets import *
import sys,pickle

from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.preprocessing import LabelEncoder

import linear_reg,svm_model,table_display,data_visualise,SVR,logistic_reg,RandomForest
import KNN,mlp,pre_trained,add_steps,gaussian


class error_window(QMainWindow):
    def __init__(self):
        super(error_window, self).__init__()
        #uic.loadUi("../ui_files/error.ui", self)
        #self.show()



class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("../ui_files/Mainwindow.ui", self)
 
        # find the widgets in the xml file
 
        #self.textedit = self.findChild(QTextEdit, "textEdit")
        #self.button = self.findChild(QPushButton, "pushButton")
        #self.button.clicked.connect(self.clickedBtn)
        global data,steps
        data=data_visualise.data_()
        steps=add_steps.add_steps()


        self.Browse = self.findChild(QPushButton,"Browse")
        self.Drop_btn = self.findChild(QPushButton,"Drop")
        
        self.fillna_btn = self.findChild(QPushButton,"fill_na")
        self.con_btn = self.findChild(QPushButton,"convert_btn")
        self.columns= self.findChild(QListWidget,"column_list")
        self.emptycolumn=self.findChild(QComboBox,"empty_column")
        self.cat_column=self.findChild(QComboBox,"cat_column")
        self.table = self.findChild(QTableView,"tableView")
        self.dropcolumns=self.findChild(QComboBox,"dropcolumn")
        self.data_shape = self.findChild(QLabel,"shape")
        self.fillmean_btn = self.findChild(QPushButton,"fillmean")
        self.submit_btn = self.findChild(QPushButton,"Submit")
        self.target_col =self.findChild(QLabel,"target_col")
        self.model_select=self.findChild(QComboBox,"model_select")
        #self.describe=self.findChild(QPlainTextEdit,"describe")
        #self.describe= self.findChild(QTextEdit,"Describe")
        
        self.scatter_x=self.findChild(QComboBox,"scatter_x")
        self.scatter_y=self.findChild(QComboBox,"scatter_y")
        self.scatter_mark=self.findChild(QComboBox,"scatter_mark")
        self.scatter_c=self.findChild(QComboBox,"scatter_c")
        self.scatter_btn = self.findChild(QPushButton,"scatterplot")
        
        self.plot_x=self.findChild(QComboBox,"plot_x")
        self.plot_y=self.findChild(QComboBox,"plot_y")
        self.plot_mark=self.findChild(QComboBox,"plot_marker")
        self.plot_c=self.findChild(QComboBox,"plot_c")
        self.plot_btn = self.findChild(QPushButton,"lineplot")

        self.hist_column=self.findChild(QComboBox,"hist_column")
        self.hist_column_add=self.findChild(QComboBox,"hist_column_add")
        self.hist_add_btn = self.findChild(QPushButton,"hist_add_btn")
        self.hist_remove_btn = self.findChild(QPushButton,"hist_remove_btn")
        self.histogram_btn = self.findChild(QPushButton,"histogram")

        self.heatmap_btn = self.findChild(QPushButton,"heatmap")

        self.columns.clicked.connect(self.target)
        self.Browse.clicked.connect(self.getCSV)
        self.Drop_btn.clicked.connect(self.dropc)
        self.scatter_btn.clicked.connect(self.scatter_plot)
        self.plot_btn.clicked.connect(self.line_plot)
        
        self.fillna_btn.clicked.connect(self.fillna)
        self.fillmean_btn.clicked.connect(self.fillme)
        
        self.hist_add_btn.clicked.connect(self.hist_add_column)
        self.hist_remove_btn.clicked.connect(self.hist_remove_column)
        self.histogram_btn.clicked.connect(self.histogram_plot)

        self.heatmap_btn.clicked.connect(self.heatmap_gen)

        self.con_btn.clicked.connect(self.con_cat)
        self.submit_btn.clicked.connect(self.set_target)

        self.train=self.findChild(QPushButton,"train")
        self.train.clicked.connect(self.train_func)
        self.scale_btn.clicked.connect(self.scale_value)
        
        self.pre_trained.clicked.connect(self.upload_model)
        self.go_pre_trained.clicked.connect(self.test_pretrained)
        self.show()

    def scale_value(self):

        #my_dict={"StandardScaler":standard_scale ,"MinMaxScaler":min_max, "PowerScaler":power_scale}
        if self.scaler.currentText()=='StandardScale':
            self.df,func_name = data.StandardScale(self.df,self.target_value)
        elif self.scaler.currentText()=='MinMaxScale':
            self.df,func_name = data.MinMaxScale(self.df,self.target_value)
        elif self.scaler.currentText()=='PowerScale':
            self.df,func_name = data.PowerScale(self.df,self.target_value)
        
        steps.add_text(self.scaler.currentText()+" applied to data")
        steps.add_pipeline(self.scaler.currentText(),func_name)
        self.filldetails()


    def hist_add_column(self):

        self.hist_column_add.addItem(self.hist_column.currentText())
        self.hist_column.removeItem(self.hist_column.findText(self.hist_column.currentText()))


    def hist_remove_column(self):
        
        self.hist_column.addItem(self.hist_column_add.currentText())
        self.hist_column_add.removeItem(self.hist_column_add.findText(self.hist_column_add.currentText()))


    def histogram_plot(self):
        
        AllItems = [self.hist_column_add.itemText(i) for i in range(self.hist_column_add.count())]
        for i in AllItems:
            data.plot_histogram(self.df,i)
        
        
    def heatmap_gen(self):

        data.plot_heatmap(self.df)

    def set_target(self):

        self.target_value=str(self.item.text()).split()[0]
        steps.add_code("target=data['"+self.target_value+"']")
        self.target_col.setText(self.target_value)

    def filldetails(self,flag=1):
         
        if(flag==0):  
            
            self.df = data.read_file(str(self.filePath))
        
        
        self.columns.clear()
        self.column_list=data.get_column_list(self.df)
        self.empty_list=data.get_empty_list(self.df)
        self.cat_col_list=data.get_cat(self.df)
        for i ,j in enumerate(self.column_list):
            stri=j+ " -------   " + str(self.df[j].dtype)
            self.columns.insertItem(i,stri)
            

        self.fill_combo_box() 
        shape_df="Shape:  Rows:"+ str(data.get_shape(self.df)[0])+"  Columns: "+str(data.get_shape(self.df)[1])
        self.data_shape.setText(shape_df)

    def fill_combo_box(self):
        
        self.dropcolumns.clear()
        self.dropcolumns.addItems(self.column_list)
        self.emptycolumn.clear()
        self.emptycolumn.addItems(self.empty_list)
        self.cat_column.clear()
        self.cat_column.addItems(self.cat_col_list)
        self.scatter_x.clear()
        self.scatter_x.addItems(self.column_list)
        self.scatter_y.clear()
        self.scatter_y.addItems(self.column_list)
        self.plot_x.clear()
        self.plot_x.addItems(self.column_list)
        self.plot_y.clear()
        self.plot_y.addItems(self.column_list)
        self.hist_column.clear()
        self.hist_column.addItems(data.get_numeric(self.df))
        self.hist_column.addItem("All")

        
        #self.describe.setText(data.get_describe(self.df))
        
        x=table_display.DataFrameModel(self.df)
        self.table.setModel(x)
        
    def upload_model(self):
        self.filePath_pre, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/akshay/Dekstop',"pkl(*.pkl)")
        with open(self.filePath_pre, 'rb') as file:
            self.pickle_model = pickle.load(file)
        
    def test_pretrained(self):

        self.testing=pre_trained.UI(self.df,self.target_value,self.pickle_model,self.filePath_pre)

    def con_cat(self):
        
        a=self.cat_column.currentText()
        self.df[a],func_name =data.convert_category(self.df,a)
        steps.add_text("Column "+ a + " converted using LabelEncoder")
        steps.add_pipeline("LabelEncoder",func_name)
        self.filldetails()

    def fillna(self):

        self.df[self.emptycolumn.currentText()]=data.fillna(self.df,self.emptycolumn.currentText())
        code="data['"+self.emptycolumn.currentText()+"'].fillna('"'Uknown'"',inplace=True)"
        steps.add_code(code)
        steps.add_text("Empty values of "+ self.emptycolumn.currentText() + " filled with Uknown")
        self.filldetails()

    def fillme(self):

        self.df[self.emptycolumn.currentText()]=data.fillmean(self.df,self.emptycolumn.currentText())
        code="data['"+column+"'].fillna(data['"+self.emptycolumn.currentText()+"'].mean(),inplace=True)"
        steps.add_code(code)
        steps.add_text("Empty values of "+ self.emptycolumn.currentText() + " filled with mean value")
        self.filldetails()

    def getCSV(self):
        self.filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/akshay/Downloads/ML Github/datasets',"csv(*.csv)")
        self.columns.clear()
        code="data=pd.read_csv('"+str(self.filePath)+"')"
        steps.add_code(code)
        steps.add_text("File "+self.filePath+" read")
        if(self.filePath!=""):
            self.filldetails(0)


    def target(self):
        self.item=self.columns.currentItem()
        
     
 
    def dropc(self):

        if (self.dropcolumns.currentText() == self.target_value):
            self.target_value=""
            self.target_col.setText("")
        self.df=data.drop_columns(self.df,self.dropcolumns.currentText())
        steps.add_code("data=data.drop('"+self.dropcolumns.currentText()+"',axis=1)")
        steps.add_text("Column "+ self.dropcolumns.currentText()+ " dropped")
        self.filldetails()  

    def scatter_plot(self):

        data.scatter_plot(df=self.df,x=self.scatter_x.currentText(),y=self.scatter_y.currentText(),c=self.scatter_c.currentText(),marker=self.scatter_mark.currentText())

        

    def line_plot(self):

        data.line_plot(df=self.df,x=self.plot_x.currentText(),y=self.plot_y.currentText(),c=self.plot_c.currentText(),marker=self.plot_mark.currentText())
     
    def train_func(self):

        myDict={ "Linear Regression":linear_reg , "SVM":svm_model ,"SVR":SVR , "Logistic Regression":logistic_reg ,"Random Forest":RandomForest,
        "K-Nearest Neighbour":KNN ,"Multi Layer Perceptron":mlp ,"Gaussian NB":gaussian}
        
        if(self.target_value!=""):
            
            self.win = myDict[self.model_select.currentText()].UI(self.df,self.target_value,steps)
            
                    
        

        


 
app = QApplication(sys.argv)
window = UI()
error_w=error_window()
app.exec_()