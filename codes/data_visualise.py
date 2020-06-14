import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,PowerTransformer
import add_steps
class data_:
	
	
	
	def read_file(self,filepath):
		
		return pd.read_csv(str(filepath))

	def convert_category(self,df,column_name):

		le=LabelEncoder()
		df[column_name] =le.fit_transform(df[column_name])
		return df[column_name],"LabelEncoder()"
	
	def get_column_list(self,df):

		column_list=[]

		for i in df.columns:
			column_list.append(i)
		return column_list

	def get_empty_list(self,df):

		empty_list=[]

		for i in df.columns:
			if(df[i].isnull().values.any()==True):
				empty_list.append(i)
		return empty_list

	def get_shape(self,df):
		return df.shape 


	def fillna(self,df,column):

		
		df[column].fillna("Uknown",inplace=True)
		return df[column]

	def fillmean(self,df,column):

		
		df[column].fillna(df[column].mean(),inplace=True)
		return df[column]

	def drop_columns(self,df,column):
		return df.drop(column,axis=1)

	def get_numeric(self,df):
		numeric_col=[]
		for i in df.columns:
			if(df[i].dtype!='object'):
				numeric_col.append(i)
		return numeric_col
	def get_cat(self,df):
		cat_col=[]
		for i in df.columns:
			if(df[i].dtype=='object'):
				cat_col.append(i)
		return cat_col

	def get_describe(self,df):

		return str(df.describe())
	
	def StandardScale(self,df,target):
		
		sc=StandardScaler()
		x=df.drop(target,axis=1)
		scaled_features=sc.fit_transform(x)
		scaled_features_df = pd.DataFrame(scaled_features, index=x.index, columns=x.columns)
		scaled_features_df[target]=df[target]
		return scaled_features_df,"StandardScaler()"

	def MinMaxScale(self,df,target):
		
		sc=MinMaxScaler()
		x=df.drop(target,axis=1)
		scaled_features=sc.fit_transform(x)
		scaled_features_df = pd.DataFrame(scaled_features, index=x.index, columns=x.columns)
		scaled_features_df[target]=df[target]
		return scaled_features_df,"MinMaxScaler()"
		
	def PowerScale(self,df,target):
		
		sc=PowerTransformer()
		x=df.drop(target,axis=1)
		scaled_features=sc.fit_transform(x)
		scaled_features_df = pd.DataFrame(scaled_features, index=x.index, columns=x.columns)
		scaled_features_df[target]=df[target]
		return scaled_features_df,"PowerTransformer()"


	def plot_histogram(self,df,column):
		
		df.hist(column=column)
		plt.show()

	def plot_heatmap(self,df):
		plt.figure()
		x=df.corr()
		mask = np.triu(np.ones_like(x, dtype=np.bool))
		sns.heatmap(x,annot=True,mask=mask,vmin=-1,vmax=1)
		plt.show()

	def scatter_plot(self,df,x,y,c,marker):
		plt.figure()
		plt.scatter(df[x],df[y],c=c,marker=marker)
		plt.xlabel(x)
		plt.ylabel(y)
		plt.title(y + " vs "+ x)
		plt.show()

	def line_plot(self,df,x,y,c,marker):
		plt.figure()
		df=df.sort_values(by=[x])
		plt.plot(df[x],df[y],c=c,marker=marker)
		plt.xlabel(x)
		plt.ylabel(y)
		plt.title(y + " vs "+ x)
		plt.show()