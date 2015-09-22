import numpy as np
import matplotlib.pyplot as plt

ux = [870, 880, 830, 780, 790, 820, 830, 803, 806, 773, 808, 811, 812, 819, 813, 828]
uy = [924, 962, 970, 980, 900, 999, 976, 984, 996, 999, 990, 1010]
uz = [900, 910, 907, 909, 911, 912, 893, 870, 922, 904, 901, 919, 909, 903, 870, 906, 912, 907, 880, 911, 913]
uw = [112, 317, 278, 88, 0, 86, 11, 70, 210, 145, 151, 320, 104, 64, 8, 65, 312, 146, 173, 141, 130, 179, 79, 215, 289, 215, 131]
ug = [193, 218, 77, 63, 82, 81, 87, 64, 72, 173, 76, 97, 75, 168, 116, 173, 204, 76, 78, 183, 110, 140, 142, 78, 212]

def analyze(data, window_size=4):
	data = np.array(data)
	
	print("Raw:", data.std())
	
	N = 10000
	means = np.empty(N)
	for i in range(N):
		selected = np.random.choice(data, size=window_size)
		means[i] = selected.mean()
	
	#plt.clf()
	#plt.hist(means, bins=50, normed=True, color="black")
	#plt.hist(data, bins=10, normed=True, alpha=0.5, color="red")
	#plt.show()
	
	print("Resampled with width %d:" % window_size, means.std())
	print("Lenght", len(data))
	
if __name__=="__main__":
	print("UX")
	analyze(ux)
	print("UY")
	analyze(uy)
	print("UZ")
	analyze(uz)
	print("UW")
	analyze(uw)
	print("UG")
	analyze(ug)