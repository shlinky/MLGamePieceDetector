import sys
import os
import random

datadirectory = ['C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/Data/ImgData/0/', 'C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/Data/ImgData/1/']
datacount = [0 for i in range(2)]
for i in datadirectory:
	for filename in os.listdir(i):
		if filename[0] == '1':
			datacount[1] += 1
		else:
			datacount[0] += 1

if datacount[0] > datacount[1]:
	more_data = 0
else:
	more_data = 1
diff = abs(datacount[0] - datacount[1])

extra_dir = datadirectory[0][:-2] + 'extras/'
os.makedirs(extra_dir)
for i in range(diff):
	imgn = random.randrange(datacount[more_data])
	img = os.listdir(datadirectory[more_data])[imgn]
	print(img)
	os.rename(datadirectory[more_data] + img, extra_dir + img)
	datacount[more_data] -= 1