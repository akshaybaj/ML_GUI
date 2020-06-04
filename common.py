
import data_visualise

class common_steps:

    def __init__ (self,df,target):

        global data 
        data=data_visualise.data_()
        self.X=df
        self.n_classes=self.X[str(target)].nunique()
        self.target_value=str(target)
        self.df=data.drop_columns(self.X,self.target_value)
        self.column_list=data.get_column_list(self.df)
    

    def return_data(self):
        return self.X ,self.n_classes ,self.target_value,self.df,self.column_list