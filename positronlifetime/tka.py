import numpy as np

class TkaFile:
	def __init__(self, filename):
		mode = None
		self.filename = filename
		self.data = []
		with open(filename, 'r') as file:
			self.live_time = float(file.readline().strip())
			self.real_time = float(file.readline().strip())
			for line in file:
				self.data.append(float(line.strip()))
		self.data = np.array(self.data)

	def __len__(self):
		return len(self.data)