# MLGamePieceDetector
Detects the relative world space position of game pieces based on image input

## Prerequisites:
### Python3:
https://docs.python.org/3/using/index.html  
pip should already be installed, but upgrade using:
```bash
python3 -m pip install --upgrade pip
```

### Tensorflow (nightly):
```bash
pip install tf-nightly
```
### Numpy:
```bash
pip install numpy
```
### Matplotlib:
```bash
pip install matplotlib
```
### Pillow:
```bash
pip install Pillow
```
## Installation:
git clone https://github.com/shlinky/MLGamePieceDetector.git  
The program needs the faster-rcnn model graph and weights to perform inference   
Download these from the Tensorflow research models github page: http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz  
You can also download one of the other faster rcnn models from the page.  
Then, place the unzipped folder inside of the localization folder.

## Demo:
To test out the program, run demo.py.
This should give a visualization of the complete algorithm working.
You can change which image it's running on by changing the image_dir variable

To use the program in the context of a larger program:  
```python
from Classification.Training.Model_Builder import ModelBuilder
from Localization.Localizer import Localizer
from World_Location_Detection.LocationCalculation import LocationCalculator
```
You can then use the three classes to classfiy, localize, and calculate world location.
