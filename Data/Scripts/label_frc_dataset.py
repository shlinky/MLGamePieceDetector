import json
import os

imgdirectory = 'C:/Users/shah-pc/Documents/Images Tagged as Valid2/Images Tagged as Valid/Filming Day 1 Images/img'
jsondirectory = 'C:/Users/shah-pc/Documents/Images Tagged as Valid2/Images Tagged as Valid/Filming Day 1 Images/ann'
datadirectory = 'C:/Users/shah-pc/Documents/Data/'

datacount = [0 for i in range(2)]
for filename in os.listdir(datadirectory):
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
	os.rename(imgf, datadirectory + name)