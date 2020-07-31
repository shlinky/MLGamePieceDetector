import json
import os

imgdirectory = 'C:/Users/shloks/Downloads/Images Tagged as Valid/Filming Day 1 Video/img'
jsondirectory = 'C:/Users/shloks/Downloads/Images Tagged as Valid/Filming Day 1 Video/ann'
datadirectory = ['C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Data/ImgData/0/', 'C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Data/ImgData/1/']

datacount = [0 for i in range(2)]
for i in datadirectory:
	for filename in os.listdir(i):
		if filename[0] == '1':
			datacount[1] += 1
		else:
			datacount[0] += 1

for filename in os.listdir(imgdirectory):
	imgf = os.path.join(imgdirectory, filename)
	jsonf = os.path.join(jsondirectory, filename + '.json')
	
	annotations = json.load(open(jsonf))
	if annotations['objects'] == []:
		piece_present = 0
	else:
		piece_present = 1

	datacount[piece_present] += 1
	name = str(piece_present) + 'data' + str(datacount[piece_present]) + '.png'
	print(datadirectory[piece_present] + name)
	os.rename(imgf, datadirectory[piece_present] + name)