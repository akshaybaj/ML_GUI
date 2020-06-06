
import matplotlib.pyplot as plt
import numpy as np

class plot_boundary:

	
	def __init__(self,x1,x2,y):

		
		plt.figure()
		self.plot_dataset(x1,x2,y,title='P2 Training set')
		


	def plot_classifier_decision(self,xx,yy,Z, mode='line'):
		if mode == 'line':
			plt.contour(xx, yy, Z)
		else:
			plt.contourf(xx, yy, Z)
		plt.xlim((np.min(x1), np.max(x1)))
		plt.lim((np.min(x2), np.max(x2)))


	def plot_dataset(self,x1,x2, y, title=None):
		
		
		plt.scatter(x1,x2, marker='o', c=y, s=25,edgecolor='k')
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.show()
		#self.plot_classifier_decision(x1,x2,y,mode='line')
	
	