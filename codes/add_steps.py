

class add_steps:
	
	def __init__(self):

		self.text=""
		

	def delete_text(self):

		self.text=""

	def add_text(self,text):

		self.text=self.text + "\n" + text
		print(text+" added")

	def save_file(self,filename):

		filename = filename.split(".")
		filename=filename[0]+".txt"
		f=open(filename, 'w')
		print(self.text)
		f.write(self.text)
